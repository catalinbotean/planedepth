"""
Download KITTI raw stereo data for Colab / constrained-disk environments.

Strategy
--------
TRAINING data  — download full ZIP for each selected drive, extract only
                 image_02/ and image_03/ (stereo colour cameras), then
                 delete the ZIP.  Two very large drives (0028 / 0034)
                 are skipped by default to save 32 GB; add them back if
                 you have the space.

TEST data      — use HTTP byte-range requests to download only the 652
                 specific frames listed in the test split, reading the
                 ZIP's central directory first and then fetching each
                 image individually (~1 GB total, vs 15+ GB full ZIPs).

Typical Colab usage
-------------------
    !python download_kitti_colab.py --out /content/kitti_data
    # or, to skip large drives and only get the test set:
    !python download_kitti_colab.py --out /content/kitti_data --test-only

    # then train:
    !python train.py --data_path /content/kitti_data --png \\
        --split eigen_zhou --model_name baseline ...

    # evaluate:
    !python evaluate_depth_HR.py --data_path /content/kitti_data --png \\
        --eval_split eigen_benchmark ...
"""

import argparse
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# Calibration zips (one per date, always small < 1 MB)
CALIB_ZIPS = [
    f"{S3}/2011_09_26_calib.zip",
    f"{S3}/2011_09_28_calib.zip",
    f"{S3}/2011_09_29_calib.zip",
    f"{S3}/2011_09_30_calib.zip",
    f"{S3}/2011_10_03_calib.zip",
]

# All training drives for the eigen_zhou split.
# The two giants (0028_sync @ 20 GB and 10_03/0034 @ 12 GB) are marked
# with their frame count so you can choose whether to include them.
TRAIN_DRIVES = [
    # (date_dir,  drive_tag,             approx frames, approx ZIP GB)
    ("2011_09_26", "2011_09_26_drive_0017_sync",   7,   0.05),
    ("2011_09_29", "2011_09_29_drive_0026_sync",   48,  0.12),
    ("2011_09_26", "2011_09_26_drive_0057_sync",   113, 0.15),
    ("2011_09_26", "2011_09_26_drive_0079_sync",   140, 0.17),
    ("2011_09_26", "2011_09_26_drive_0113_sync",   151, 0.20),
    ("2011_09_28", "2011_09_28_drive_0001_sync",   161, 0.21),
    ("2011_09_26", "2011_09_26_drive_0001_sync",   189, 0.44),
    ("2011_09_26", "2011_09_26_drive_0018_sync",   198, 0.31),
    ("2011_09_26", "2011_09_26_drive_0011_sync",   210, 0.30),
    ("2011_09_26", "2011_09_26_drive_0035_sync",   233, 0.35),
    ("2011_09_26", "2011_09_26_drive_0005_sync",   263, 0.37),
    ("2011_09_26", "2011_09_26_drive_0095_sync",   476, 0.55),
    ("2011_09_26", "2011_09_26_drive_0015_sync",   535, 0.62),
    ("2011_09_26", "2011_09_26_drive_0051_sync",   547, 0.62),
    ("2011_09_26", "2011_09_26_drive_0014_sync",   549, 0.63),
    ("2011_09_29", "2011_09_29_drive_0004_sync",   570, 0.66),
    ("2011_09_26", "2011_09_26_drive_0104_sync",   565, 0.65),
    ("2011_09_26", "2011_09_26_drive_0091_sync",   594, 0.69),
    ("2011_09_26", "2011_09_26_drive_0032_sync",   700, 0.81),
    ("2011_09_26", "2011_09_26_drive_0039_sync",   701, 0.82),
    ("2011_09_26", "2011_09_26_drive_0070_sync",   757, 0.86),
    ("2011_09_26", "2011_09_26_drive_0019_sync",   769, 0.88),
    ("2011_09_26", "2011_09_26_drive_0028_sync",   783, 0.89),
    ("2011_09_26", "2011_09_26_drive_0061_sync",  1268, 1.46),
    ("2011_09_26", "2011_09_26_drive_0087_sync",  1296, 1.50),
    ("2011_09_26", "2011_09_26_drive_0022_sync",  1427, 3.25),
    ("2011_09_30", "2011_09_30_drive_0020_sync",  1984, 2.30),
    ("2011_10_03", "2011_10_03_drive_0042_sync",  1994, 2.31),
    ("2011_09_30", "2011_09_30_drive_0034_sync",  2009, 2.33),
    ("2011_09_30", "2011_09_30_drive_0033_sync",  2873, 3.33),
    # ── Large drives: skip with --skip-large (saves ~32 GB) ──────────────
    ("2011_09_30", "2011_09_30_drive_0028_sync",  9287, 20.2),  # LARGE
    ("2011_10_03", "2011_10_03_drive_0034_sync",  8413, 12.0),  # LARGE
]

# Drives used ONLY for evaluation (not in training split)
EVAL_DRIVES = [
    ("2011_09_26", "2011_09_26_drive_0002_sync"),
    ("2011_09_26", "2011_09_26_drive_0009_sync"),
    ("2011_09_26", "2011_09_26_drive_0013_sync"),
    ("2011_09_26", "2011_09_26_drive_0020_sync"),
    ("2011_09_26", "2011_09_26_drive_0023_sync"),
    ("2011_09_26", "2011_09_26_drive_0027_sync"),
    ("2011_09_26", "2011_09_26_drive_0029_sync"),
    ("2011_09_26", "2011_09_26_drive_0036_sync"),
    ("2011_09_26", "2011_09_26_drive_0046_sync"),
    ("2011_09_26", "2011_09_26_drive_0048_sync"),
    ("2011_09_26", "2011_09_26_drive_0052_sync"),
    ("2011_09_26", "2011_09_26_drive_0056_sync"),
    ("2011_09_26", "2011_09_26_drive_0059_sync"),
    ("2011_09_26", "2011_09_26_drive_0064_sync"),
    ("2011_09_26", "2011_09_26_drive_0084_sync"),
    ("2011_09_26", "2011_09_26_drive_0086_sync"),
    ("2011_09_26", "2011_09_26_drive_0093_sync"),
    ("2011_09_26", "2011_09_26_drive_0096_sync"),
    ("2011_09_26", "2011_09_26_drive_0101_sync"),
    ("2011_09_26", "2011_09_26_drive_0106_sync"),
    ("2011_09_26", "2011_09_26_drive_0117_sync"),
    ("2011_09_28", "2011_09_28_drive_0002_sync"),
    ("2011_09_29", "2011_09_29_drive_0071_sync"),
    ("2011_09_30", "2011_09_30_drive_0016_sync"),
    ("2011_09_30", "2011_09_30_drive_0018_sync"),
    ("2011_09_30", "2011_09_30_drive_0027_sync"),
    ("2011_10_03", "2011_10_03_drive_0027_sync"),
    ("2011_10_03", "2011_10_03_drive_0047_sync"),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _zip_url(date_dir, drive_tag):
    drive_num = drive_tag.split("_drive_")[1].replace("_sync", "")
    return f"{S3}/{date_dir}_drive_{drive_num}/{drive_tag}.zip"


def _http_get(url, byte_range=None, retries=4):
    headers = {}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} after {wait}s: {e}")
            time.sleep(wait)


def _progress(done, total, prefix=""):
    bar = int(40 * done / max(total, 1))
    pct = 100 * done / max(total, 1)
    sys.stdout.write(f"\r{prefix}[{'#'*bar}{'.'*(40-bar)}] {pct:.0f}%")
    sys.stdout.flush()


# ── Central-directory parser ──────────────────────────────────────────────────

def _parse_cd(data):
    """Parse ZIP central directory; return {filename: (local_offset, comp_size)}."""
    entries = {}
    i = 0
    while i < len(data) - 46:
        if data[i:i+4] != b'PK\x01\x02':
            i += 1
            continue
        (ver_made, ver_needed, flag, method,
         mod_time, mod_date, crc,
         comp_size, uncomp_size,
         fname_len, extra_len, comment_len,
         disk_num, int_attr) = struct.unpack_from('<HHHHHHI II HHH HH', data, i+4)
        ext_attr, local_offset = struct.unpack_from('<II', data, i+38)
        fname = data[i+46:i+46+fname_len].decode('utf-8', errors='replace')
        if comp_size > 0:
            entries[fname] = (local_offset, comp_size)
        i += 46 + fname_len + extra_len + comment_len
    return entries


def _fetch_cd(url):
    """Fetch central directory of a remote ZIP without downloading the whole file."""
    # Get file size via HEAD
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers['Content-Length'])

    # Read last 65 KB — enough for the EOCD + central directory of KITTI ZIPs
    tail_size = min(65536, total)
    tail = _http_get(url, (total - tail_size, total - 1))

    # Find EOCD signature
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd == -1:
        raise RuntimeError(f"No EOCD in {url}")

    cd_size, cd_offset = struct.unpack_from('<II', tail, eocd + 12)
    if cd_offset + cd_size > total - tail_size + eocd:
        # Central directory starts before our tail — fetch more
        cd_data = _http_get(url, (cd_offset, cd_offset + cd_size - 1))
    else:
        cd_start = cd_offset - (total - tail_size)
        cd_data = tail[cd_start: cd_start + cd_size]

    return _parse_cd(cd_data), total


def _extract_png_from_local_hdr(raw, expected_size):
    """Given raw bytes starting at a ZIP local file header, return the PNG data."""
    if raw[:4] != b'PK\x03\x04':
        png_start = raw.find(b'\x89PNG')
        return raw[png_start:png_start + expected_size] if png_start != -1 else None
    fname_len, extra_len = struct.unpack_from('<HH', raw, 26)
    hdr_size = 30 + fname_len + extra_len
    blob = raw[hdr_size:]
    png_start = blob.find(b'\x89PNG')
    return blob[png_start:png_start + expected_size] if png_start != -1 else blob[:expected_size]


# ── Calibration download ──────────────────────────────────────────────────────

def download_calibration(out_dir):
    for url in CALIB_ZIPS:
        date = os.path.basename(url).replace('_calib.zip', '')
        dest = os.path.join(out_dir, date)
        os.makedirs(dest, exist_ok=True)
        # Check if already extracted
        if os.path.exists(os.path.join(dest, 'calib_cam_to_cam.txt')):
            print(f"  calib {date}: already present, skipping")
            continue
        print(f"  calib {date}: downloading ...", end=' ')
        data = _http_get(url)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base.startswith('calib_') and base.endswith('.txt'):
                    target = os.path.join(dest, base)
                    with open(target, 'wb') as f:
                        f.write(zf.read(name))
        print("done")


# ── Full-ZIP training download ────────────────────────────────────────────────

def download_train_drive(date_dir, drive_tag, out_dir):
    """Download ZIP, extract image_02+03, delete ZIP. Returns frame count."""
    url      = _zip_url(date_dir, drive_tag)
    zip_path = f"/tmp/_kitti_{drive_tag}.zip"
    drive_out = os.path.join(out_dir, date_dir, drive_tag)

    # Skip if already extracted
    im02 = os.path.join(drive_out, 'image_02', 'data')
    if os.path.isdir(im02) and len(os.listdir(im02)) > 0:
        n = len(os.listdir(im02))
        print(f"  {drive_tag}: already present ({n} frames), skipping")
        return n

    print(f"  {drive_tag}: downloading ZIP ...")
    # Stream-download with progress
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get('Content-Length', 0))
        done  = 0
        chunk = 1 << 20   # 1 MB
        with open(zip_path, 'wb') as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                _progress(done, total, f"  {drive_tag}: ")
    print()

    print(f"  {drive_tag}: extracting image_02 + image_03 ...")
    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if '/image_02/' not in name and '/image_03/' not in name:
                continue
            rel  = name.split(f'{drive_tag}/', 1)[-1]          # image_02/data/0...png
            dest = os.path.join(drive_out, rel)
            if info.is_dir():
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(info) as src, open(dest, 'wb') as dst:
                    dst.write(src.read())
                n_files += 1

    os.remove(zip_path)
    print(f"  {drive_tag}: extracted {n_files} files, ZIP deleted")
    return n_files


# ── Range-request eval download ───────────────────────────────────────────────

def download_eval_drive(date_dir, drive_tag, out_dir):
    """Download only the PNG files needed for evaluation via range requests."""
    url = _zip_url(date_dir, drive_tag)
    drive_out = os.path.join(out_dir, date_dir, drive_tag)

    # Check if already done
    im02 = os.path.join(drive_out, 'image_02', 'data')
    if os.path.isdir(im02) and len(os.listdir(im02)) > 0:
        print(f"  {drive_tag}: already present, skipping")
        return

    print(f"  {drive_tag}: fetching central directory ...", end=' ', flush=True)
    try:
        cd, zip_total = _fetch_cd(url)
    except Exception as e:
        print(f"FAILED ({e})")
        return
    print(f"{len(cd)} entries")

    # Filter to image_02 and image_03 PNGs
    img_entries = {
        k: v for k, v in cd.items()
        if ('/image_02/' in k or '/image_03/' in k) and k.endswith('.png')
    }
    print(f"  {drive_tag}: fetching {len(img_entries)} images via range requests ...")

    fetched = 0
    for fname, (local_off, comp_size) in sorted(img_entries.items()):
        # Derive output path: strip the leading date_dir/drive_tag/
        rel = fname.split(f'{drive_tag}/', 1)[-1]
        dest = os.path.join(drive_out, rel)
        if os.path.exists(dest):
            fetched += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        end_byte = local_off + comp_size + 64   # +64 for local header overhead
        raw = _http_get(url, (local_off, min(end_byte, zip_total - 1)))
        png = _extract_png_from_local_hdr(raw, comp_size)
        if png and png[:4] == b'\x89PNG':
            with open(dest, 'wb') as f:
                f.write(png)
            fetched += 1
        else:
            print(f"\n    WARNING: could not extract {fname}")

        _progress(fetched, len(img_entries), f"  {drive_tag}: ")

    print(f"\n  {drive_tag}: {fetched}/{len(img_entries)} images downloaded")


# ── Mini split generator ──────────────────────────────────────────────────────

def write_mini_split(out_dir, train_drives_downloaded):
    """Create splits/eigen_zhou_mini/ with train/val files for downloaded drives only."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir    = os.path.join(script_dir, 'splits', 'eigen_zhou')
    dst_dir    = os.path.join(script_dir, 'splits', 'eigen_zhou_mini')
    os.makedirs(dst_dir, exist_ok=True)

    downloaded = {row[1] for row in train_drives_downloaded}  # set of drive_tags

    # Filter train_files.txt
    kept = []
    with open(os.path.join(src_dir, 'train_files.txt')) as f:
        for line in f:
            drive = line.split()[0].split('/')[-1]  # last path component
            if drive in downloaded:
                kept.append(line)
    with open(os.path.join(dst_dir, 'train_files.txt'), 'w') as f:
        f.writelines(kept)

    # Copy val_files.txt unchanged (val drives overlap with some training drives)
    import shutil
    shutil.copy(os.path.join(src_dir, 'val_files.txt'),
                os.path.join(dst_dir, 'val_files.txt'))

    print(f"\nMini split written to splits/eigen_zhou_mini/")
    print(f"  train: {len(kept)} frames from {len(downloaded)} drives")
    print("  Use --split eigen_zhou_mini when training")
    return dst_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download KITTI stereo data efficiently for Colab / limited disk")
    parser.add_argument("--out",         default="./kitti_data",
                        help="output directory (default: ./kitti_data)")
    parser.add_argument("--test-only",   action="store_true",
                        help="only download evaluation drives, skip training data")
    parser.add_argument("--skip-large",  action="store_true", default=True,
                        help="skip the 2 large training drives "
                             "(0028_sync @ 20 GB and 10_03/0034 @ 12 GB) "
                             "[default: True]")
    parser.add_argument("--include-large", action="store_true",
                        help="include the large drives (overrides --skip-large)")
    parser.add_argument("--max-gb",      type=float, default=50.0,
                        help="stop downloading training ZIPs after this many GB "
                             "(default: 50)")
    args = parser.parse_args()

    skip_large = args.skip_large and not args.include_large
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # ── Calibration ──────────────────────────────────────────────────────────
    print("=== Calibration files ===")
    download_calibration(out)

    # ── Training data ─────────────────────────────────────────────────────────
    downloaded_train = []
    if not args.test_only:
        print("\n=== Training drives ===")
        total_gb = 0.0
        for row in TRAIN_DRIVES:
            date_dir, drive_tag, frames, approx_gb = row
            if skip_large and approx_gb > 10.0:
                print(f"  {drive_tag}: SKIPPED (large, {approx_gb:.0f} GB "
                      f"— use --include-large to enable)")
                continue
            if total_gb + approx_gb > args.max_gb:
                print(f"  stopping at {total_gb:.1f} GB (--max-gb={args.max_gb})")
                break
            download_train_drive(date_dir, drive_tag, out)
            downloaded_train.append(row)
            total_gb += approx_gb

        print(f"\nTraining: {sum(r[2] for r in downloaded_train)} frames "
              f"in {len(downloaded_train)} drives "
              f"({sum(r[3] for r in downloaded_train):.1f} GB)")
        write_mini_split(out, downloaded_train)

    # ── Evaluation data ───────────────────────────────────────────────────────
    print("\n=== Evaluation drives (range-request, ~1 GB total) ===")
    for date_dir, drive_tag in EVAL_DRIVES:
        download_eval_drive(date_dir, drive_tag, out)

    print("\n=== Done ===")
    print(f"Data at: {out}")
    print()
    print("Training command (mini subset):")
    print("  python train.py \\")
    print(f"    --data_path {out} --png \\")
    print("    --split eigen_zhou_mini \\")
    print("    --model_name baseline_mini \\")
    print("    --use_mixture_loss --use_denseaspp --plane_residual \\")
    print("    --num_epochs 20 --batch_size 8")
    print()
    print("Evaluation command:")
    print("  python evaluate_depth_HR.py \\")
    print(f"    --data_path {out} --png \\")
    print("    --eval_split eigen_benchmark \\")
    print("    --load_weights_folder ./log/baseline_mini/models/weights_19")


if __name__ == "__main__":
    main()
