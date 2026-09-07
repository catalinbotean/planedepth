"""Colour KITTI ground-truth depth PNGs the way the predictions are coloured.

The annotated maps are 16-bit, depth = value / 256 metres, zero where there is
no measurement. They are converted to disparity, normalised on the same 5th/95th
percentiles the prediction panels use, and written as 8-bit grey for ffmpeg's
magma pseudocolour; holes stay at zero, which magma renders black.

Usage:
    python scripts/colorize_gt.py <out_dir> <gt.png> [<gt.png> ...] [--cmap turbo]

--cmap takes any ffmpeg pseudocolor preset (magma, inferno, plasma, viridis,
turbo, cividis); it must match the one the predictions use.
"""
import os
import struct
import subprocess
import sys
import zlib


def read_gray16(path):
    probe = subprocess.run(["ffprobe", "-loglevel", "error", "-select_streams", "v",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            path], capture_output=True, text=True).stdout.strip()
    w, h = (int(v) for v in probe.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
    return w, h, struct.unpack("<{}H".format(w * h), raw[:w * h * 2])


def colormap(preset, samples=256):
    """The preset's lookup table, read back from ffmpeg on a grey ramp.

    ffmpeg owns the colour definitions; sampling them here means the ground
    truth is coloured with exactly the same table as everything else, while the
    mapping itself stays in Python, where holes can be left truly black.
    """
    ramp = bytes(range(samples))
    write_gray8("/tmp/_ramp.png", samples, 1, ramp)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "/tmp/_ramp.png",
                          "-vf", "format=yuv444p,pseudocolor=preset={}".format(preset),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    os.remove("/tmp/_ramp.png")
    return [tuple(raw[i * 3:i * 3 + 3]) for i in range(samples)]


def dilate(values, w, h, rounds=2):
    """Grow the measured points so single LiDAR returns survive downscaling."""
    for _ in range(rounds):
        out = list(values)
        for y in range(h):
            row = y * w
            for x in range(w):
                i = row + x
                if values[i]:
                    continue
                best = 0
                for dy in (-1, 0, 1):
                    yy = y + dy
                    if yy < 0 or yy >= h:
                        continue
                    for dx in (-1, 0, 1):
                        xx = x + dx
                        if 0 <= xx < w:
                            v = values[yy * w + xx]
                            if v > best:
                                best = v
                out[i] = best
        values = out
    return values


def write_rgb8(path, w, h, pixels):
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))

    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            rows += bytes(pixels[y * w + x])
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(rows), 6)))
        f.write(chunk(b"IEND", b""))


def write_gray8(path, w, h, data):
    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))

    rows = b"".join(b"\x00" + bytes(data[y * w:(y + 1) * w]) for y in range(h))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(rows, 6)))
        f.write(chunk(b"IEND", b""))


def main():
    argv = sys.argv[1:]
    cmap = "magma"
    if "--cmap" in argv:
        i = argv.index("--cmap")
        cmap = argv[i + 1]
        del argv[i:i + 2]
    out_dir, paths = argv[0], argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        w, h, values = read_gray16(path)
        disp = [0.0 if v == 0 else 256.0 / v for v in values]
        valid = sorted(d for d in disp if d > 0)
        if not valid:
            print("   no measurements in " + path)
            continue
        lo = valid[int(0.05 * (len(valid) - 1))]
        hi = valid[int(0.95 * (len(valid) - 1))]
        span = (hi - lo) or 1e-6

        grey = [0] * (w * h)
        for i, d in enumerate(disp):
            if d > 0:
                t = (d - lo) / span
                grey[i] = 1 + int(254 * (0.0 if t < 0 else 1.0 if t > 1 else t))
        grey = dilate(grey, w, h)

        lut = colormap(cmap)
        pixels = [(0, 0, 0) if v == 0 else lut[v] for v in grey]

        stem = os.path.basename(path)
        stem = stem[:-len("_gt.png")] if stem.endswith("_gt.png") else os.path.splitext(stem)[0]
        out = os.path.join(out_dir, stem + "_gtcolor.png")
        write_rgb8(out, w, h, pixels)
        print("   {}  ({}x{}, {:.1f}% measured)".format(
            out, w, h, 100.0 * len(valid) / (w * h)))


if __name__ == "__main__":
    main()
