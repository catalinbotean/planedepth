"""Fetch the annotated KITTI ground-truth depth of a few frames, over HTTP.

data_depth_annotated.zip is 14 GB, but only a handful of frames are ever needed
for a figure. The archive's central directory is read with byte-range requests,
the wanted entries are located in it, and only their compressed bytes are
downloaded and inflated - a few hundred kilobytes instead of the whole file.

Usage:
    python scripts/fetch_kitti_gt.py <out_dir> <stem> [<stem> ...]

where a stem is the name the inference and evaluation scripts already use,
e.g. 2011_09_26_drive_0013_sync_0000000096. Frames the annotated set does not
cover are reported and skipped.
"""
import os
import re
import struct
import subprocess
import sys
import zlib

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
STEM = re.compile(r"^(.*_sync)_(\d{10})$")


def fetch(start, end):
    """One byte range, through curl: the stock python here has no CA bundle."""
    out = subprocess.run(["curl", "-sf", "-H", "Range: bytes={}-{}".format(start, end), URL],
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit("range request {}-{} failed".format(start, end))
    return out.stdout


def total_size():
    head = subprocess.run(["curl", "-sfI", URL], capture_output=True, text=True).stdout
    for line in head.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1])
    raise SystemExit("could not read the archive size")


def central_directory(size):
    tail = fetch(size - 65536, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise SystemExit("no end-of-central-directory record")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])

    locator = tail.rfind(b"PK\x06\x07")
    if cd_off == 0xFFFFFFFF or locator >= 0:      # zip64
        z64_off = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        head = fetch(z64_off, z64_off + 56)
        cd_size, cd_off = struct.unpack("<QQ", head[40:56])
    return fetch(cd_off, cd_off + cd_size - 1)


def entries(cd):
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos:pos + 4] != b"PK\x01\x02":
            break
        method, = struct.unpack("<H", cd[pos + 10:pos + 12])
        csize, usize = struct.unpack("<II", cd[pos + 20:pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        local, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, local):    # zip64 extra field
            i = 0
            while i + 4 <= len(extra):
                tag, ln = struct.unpack("<HH", extra[i:i + 4])
                if tag == 0x0001:
                    vals = struct.unpack("<{}Q".format(ln // 8), extra[i + 4:i + 4 + ln])
                    it = iter(vals)
                    if usize == 0xFFFFFFFF:
                        usize = next(it)
                    if csize == 0xFFFFFFFF:
                        csize = next(it)
                    if local == 0xFFFFFFFF:
                        local = next(it)
                    break
                i += 4 + ln
        yield name, method, csize, local
        pos += 46 + nlen + elen + clen


def main():
    out_dir, stems = sys.argv[1], sys.argv[2:]
    if not stems:
        raise SystemExit(__doc__)
    os.makedirs(out_dir, exist_ok=True)

    wanted = {}
    for stem in stems:
        m = STEM.match(stem)
        if not m:
            raise SystemExit("not a frame stem: " + stem)
        drive, frame = m.group(1), m.group(2)
        # the annotated set splits drives into train/ and val/
        for split in ("train", "val"):
            key = "{}/{}/proj_depth/groundtruth/image_02/{}.png".format(split, drive, frame)
            wanted[key] = stem

    print("-> reading the central directory of a {} byte archive".format(total_size()))
    cd = central_directory(total_size())

    found = {}
    for name, method, csize, local in entries(cd):
        if name in wanted:
            found[wanted[name]] = (name, method, csize, local)

    for stem in stems:
        if stem not in found:
            print("   not in the annotated set: {}".format(stem))
            continue
        name, method, csize, local = found[stem]
        header = fetch(local, local + 29)
        nlen, elen = struct.unpack("<HH", header[26:30])
        data_at = local + 30 + nlen + elen
        blob = fetch(data_at, data_at + csize - 1)
        raw = zlib.decompress(blob, -15) if method == 8 else blob
        path = os.path.join(out_dir, stem + "_gt.png")
        with open(path, "wb") as f:
            f.write(raw)
        print("   {}  <-  {} ({} bytes)".format(path, name, len(raw)))


if __name__ == "__main__":
    main()
