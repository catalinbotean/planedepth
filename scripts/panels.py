"""Shared helpers for reading the panels written by --eval_out_dir.

A panel stacks three strips of equal height - input, prediction, ground truth -
separated by two 8 px gutters, at whatever resolution the model was evaluated
at. Deriving the geometry from the file keeps the analysis scripts independent
of the dataset and of the evaluation resolution.
"""
import subprocess


def size_of(path):
    out = subprocess.run(["ffprobe", "-loglevel", "error", "-select_streams", "v",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          path], capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def strip_count(path, gutter=8):
    """How many strips a panel holds: 3 with ground truth, 2 without it.

    The evaluation writes input/prediction/ground truth; inference has no
    ground truth and writes input/prediction. Only one of the two layouts
    divides evenly for a given panel height, which is what distinguishes them.
    """
    _, h = size_of(path)
    if (h - 2 * gutter) % 3 == 0:
        return 3
    if (h - gutter) % 2 == 0:
        return 2
    raise SystemExit("{} is not a 2- or 3-strip panel".format(path))


def strips(path, gutter=8):
    """The crop expressions of a panel, in order, starting with the input."""
    w, h = size_of(path)
    n = strip_count(path, gutter)
    sh = (h - (n - 1) * gutter) // n
    return ["{}:{}:0:{}".format(w, sh, i * (sh + gutter)) for i in range(n)]


def gray(path, crop, w, h, nearest=False):
    """Decode a cropped, downscaled strip as a flat list of luminance values."""
    flags = ":flags=neighbor" if nearest else ""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-vf",
         "crop={},scale={}:{}{}".format(crop, w, h, flags), "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return list(out[:w * h])
