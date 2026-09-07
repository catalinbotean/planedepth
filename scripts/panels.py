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


def strips(path, gutter=8):
    """The three crop expressions of a panel, as (input, prediction, gt)."""
    w, h = size_of(path)
    sh = (h - 2 * gutter) // 3
    return ["{}:{}:0:{}".format(w, sh, 0),
            "{}:{}:0:{}".format(w, sh, sh + gutter),
            "{}:{}:0:{}".format(w, sh, 2 * (sh + gutter))]


def gray(path, crop, w, h, nearest=False):
    """Decode a cropped, downscaled strip as a flat list of luminance values."""
    flags = ":flags=neighbor" if nearest else ""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-vf",
         "crop={},scale={}:{}{}".format(crop, w, h, flags), "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return list(out[:w * h])
