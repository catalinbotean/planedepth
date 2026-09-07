"""Error maps against the KITTI ground truth, on a shared error scale.

Reads the metric depth written by infer.py / infer_baseline.py (--save_npy) and
the annotated ground truth fetched by scripts/fetch_kitti_gt.py, and writes one
coloured absolute-error map per frame and model. Pixels without a laser
measurement are left black; the error scale is fixed across every model and
frame, which is what makes the maps comparable.

Usage:
    python scripts/error_maps.py <fig4_dir> <gt_dir> <out_dir> [--max_error 5]
        [--models monodepth2,planedepth,scopedepth] [--cmap inferno]
"""
import os
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_gt import colormap, dilate, write_rgb8


def read_npy(path):
    """Minimal .npy reader: no numpy on the machine that builds the figures."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise SystemExit("not a .npy file: " + path)
        major = f.read(2)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I",
                             f.read(2 if major == 1 else 4))[0]
        header = eval(f.read(hlen).decode("latin1"))          # a plain dict literal
        h, w = header["shape"]
        dtype = header["descr"]
        code = {"<f4": ("f", 4), "<f8": ("d", 8)}.get(dtype)
        if code is None:
            raise SystemExit("unsupported dtype {} in {}".format(dtype, path))
        fmt, size = code
        data = f.read(w * h * size)
        return w, h, struct.unpack("<{}{}".format(w * h, fmt), data)


def read_gt(path):
    probe = subprocess.run(["ffprobe", "-loglevel", "error", "-select_streams", "v",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            path], capture_output=True, text=True).stdout.strip()
    w, h = (int(v) for v in probe.split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
    values = struct.unpack("<{}H".format(w * h), raw[:w * h * 2])
    return w, h, [v / 256.0 for v in values]          # metres, 0 where unmeasured


def main():
    argv = sys.argv[1:]
    def option(name, default):
        if name in argv:
            i = argv.index(name)
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        return default

    max_error = float(option("--max_error", "5"))
    models = option("--models", "monodepth2,planedepth,scopedepth").split(",")
    cmap = option("--cmap", "inferno")
    fig_dir, gt_dir, out_dir = argv[0], argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)
    lut = colormap(cmap)

    stems = sorted(f[:-len("_gt.png")] for f in os.listdir(gt_dir)
                   if f.endswith("_gt.png"))
    print("{:<40} {}".format("scene", "  ".join("{:>12}".format(m) for m in models)))
    for stem in stems:
        gw, gh, gt = read_gt(os.path.join(gt_dir, stem + "_gt.png"))
        means = []
        for model in models:
            npy = os.path.join(fig_dir, model, stem + "_depth.npy")
            if not os.path.isfile(npy):
                means.append(float("nan"))
                continue
            pw, ph, pred = read_npy(npy)
            if (pw, ph) != (gw, gh):
                raise SystemExit("{} is {}x{}, ground truth is {}x{}".format(
                    npy, pw, ph, gw, gh))

            grey = [0] * (gw * gh)
            total, count = 0.0, 0
            for i, g in enumerate(gt):
                if g <= 0:
                    continue
                e = abs(pred[i] - g)
                total += e
                count += 1
                t = min(e / max_error, 1.0)
                grey[i] = 1 + int(254 * t)
            means.append(total / max(count, 1))

            grey = dilate(grey, gw, gh)
            write_rgb8(os.path.join(out_dir, "{}_{}_err.png".format(stem, model)),
                       gw, gh, [(0, 0, 0) if v == 0 else lut[v] for v in grey])
        print("{:<40} {}".format(
            stem, "  ".join("{:12.3f}".format(m) for m in means)))

    print("\nmean absolute error in metres over the measured pixels; "
          "maps are clipped at {} m".format(max_error))


if __name__ == "__main__":
    main()
