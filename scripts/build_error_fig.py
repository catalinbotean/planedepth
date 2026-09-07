"""Assemble the figure the reviewer asked for: a shared depth scale and error maps.

Each scene becomes a block of two columns - the depth map on the left, the
absolute error against the laser ground truth on the right - with the input and
the ground truth on the first row. Both scales are fixed across every model and
every scene, and colour bars at the foot of the figure give the units.

Usage:
    python scripts/build_error_fig.py <fig4_dir> <gt_color_dir> <err_dir> <out.png> \
        --scene "<stem>|(a) caption" [--scene ...] \
        [--models monodepth2,planedepth,scopedepth] \
        [--labels "Monodepth2,PlaneDepth,SCOPE-Depth (ours)"] \
        [--depth_range 3 80] [--max_error 5] [--depth_cmap turbo] [--err_cmap inferno]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorize_gt import colormap, write_rgb8
from panels import size_of
from textimg import render

GUT, BLOCK_GAP, ROW_GAP = 6, 40, 54


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + args, check=True)


def solid(work, w, h, colour="white"):
    path = os.path.join(work, "spacer_{}_{}x{}.png".format(colour, w, h))
    if not os.path.isfile(path):
        run(["-f", "lavfi", "-i", "color={}:s={}x{}".format(colour, w, h),
             "-frames:v", "1", path])
    return path


def badge(work, text, height):
    key = "".join(c for c in text if c.isalnum()) or "x"
    path = os.path.join(work, "badge_" + key + ".png")
    if not os.path.isfile(path):
        render("<span style=\"display:inline-block;background:#151515;color:#fff;"
               "font-family:Helvetica,Arial,sans-serif;font-size:13px;"
               "font-weight:600;padding:3px 7px\">{}</span>".format(text),
               path, height)
    return path


def tile(work, src, label, w, h, out):
    run(["-i", src, "-i", badge(work, label, max(26, h // 9)), "-filter_complex",
         "[0]scale={}:{}[p];[p][1]overlay=16:14".format(w, h), out])
    return out


def colour_bar(work, cmap, low_text, high_text, width, height, unit, out):
    """A horizontal ramp of the colormap, with its two end values on it."""
    lut = colormap(cmap)
    pixels = [lut[min(255, x * 256 // width)] for _ in range(height) for x in range(width)]
    ramp = os.path.join(work, "ramp_{}.png".format(cmap))
    write_rgb8(ramp, width, height, pixels)

    # the ends of a colormap are dark or light unpredictably, so the labels ride
    # in the same dark badge the panels use
    low = badge(work, low_text, height - 14)
    high = badge(work, high_text, height - 14)
    run(["-i", ramp, "-i", low, "-i", high, "-filter_complex",
         "[0][1]overlay=10:(H-h)/2[a];[a][2]overlay=W-w-10:(H-h)/2", out])

    title = os.path.join(work, "cb_title_{}.png".format(cmap))
    render("<span style=\"font-family:Palatino,Georgia,serif;font-size:12px;"
           "color:#111\">{}</span>".format(unit), title, max(20, height - 12))
    return out, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig_dir")
    ap.add_argument("gt_dir")
    ap.add_argument("err_dir")
    ap.add_argument("out")
    ap.add_argument("--scene", action="append", required=True, help="stem|caption")
    ap.add_argument("--models", default="monodepth2,planedepth,scopedepth")
    ap.add_argument("--labels", default="Monodepth2,PlaneDepth,SCOPE-Depth (ours)")
    ap.add_argument("--depth_range", nargs=2, default=["3", "80"])
    ap.add_argument("--max_error", default="5")
    ap.add_argument("--depth_cmap", default="turbo")
    ap.add_argument("--err_cmap", default="inferno")
    args = ap.parse_args()

    models = args.models.split(",")
    labels = args.labels.split(",")
    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_err_work")
    os.makedirs(work, exist_ok=True)

    scenes = [s.split("|", 1) for s in args.scene]
    first = os.path.join(args.fig_dir, models[0], scenes[0][0] + "_pred.png")
    w, h = size_of(first)

    blocks = []
    for stem, caption in scenes:
        safe = stem.replace(".", "_")
        rows = []

        cap_raw = render("<span style=\"font-family:Palatino,Georgia,serif;"
                         "font-size:14px;color:#111\">{}</span>".format(caption),
                         os.path.join(work, safe + "_capraw.png"), max(26, h // 9))
        cap = os.path.join(work, safe + "_cap.png")
        run(["-i", os.path.join(work, safe + "_capraw.png"), "-vf",
             "pad={}:{}:0:0:white".format(2 * w + GUT, h // 7 + 10), cap])
        rows.append(cap)

        pairs = [(os.path.join(args.fig_dir, models[0], stem + "_panel.png"), "Input",
                  os.path.join(args.gt_dir, stem + "_gtcolor.png"), "Ground truth")]
        for model, label in zip(models, labels):
            pairs.append((os.path.join(args.fig_dir, model, stem + "_pred.png"), label,
                          os.path.join(args.err_dir, "{}_{}_err.png".format(stem, model)),
                          label + " error"))

        for i, (left_src, left_label, right_src, right_label) in enumerate(pairs):
            if i == 0:      # the input lives in the top strip of the panel
                cropped = os.path.join(work, "{}_input.png".format(safe))
                run(["-i", left_src, "-vf", "crop={}:{}:0:0".format(w, h), cropped])
                left_src = cropped
            lt = tile(work, left_src, left_label, w, h, os.path.join(work, "{}_{}_l.png".format(safe, i)))
            rt = tile(work, right_src, right_label, w, h, os.path.join(work, "{}_{}_r.png".format(safe, i)))
            row = os.path.join(work, "{}_{}_row.png".format(safe, i))
            run(["-i", lt, "-i", solid(work, GUT, h), "-i", rt,
                 "-filter_complex", "hstack=inputs=3", row])
            rows.append(row)
            if i < len(pairs) - 1:
                rows.append(solid(work, 2 * w + GUT, GUT))

        block = os.path.join(work, "block_" + safe + ".png")
        run([a for p in rows for a in ("-i", p)] +
            ["-filter_complex", "vstack=inputs={}".format(len(rows)), block])
        blocks.append(block)

    bw, bh = size_of(blocks[0])
    grid_rows = []
    for i in range(0, len(blocks), 2):
        pair = blocks[i:i + 2]
        row = os.path.join(work, "grid{}.png".format(i))
        if len(pair) == 2:
            run(["-i", pair[0], "-i", solid(work, BLOCK_GAP, bh), "-i", pair[1],
                 "-filter_complex", "hstack=inputs=3", row])
        else:
            run(["-i", pair[0], "-vf", "pad={}:{}:0:0:white".format(2 * bw + BLOCK_GAP, bh), row])
        grid_rows.append(row)

    rw, _ = size_of(grid_rows[0])
    bar_w = (rw - 3 * BLOCK_GAP) // 2
    depth_bar, depth_title = colour_bar(
        work, args.depth_cmap, "{} m".format(args.depth_range[1]),
        "{} m".format(args.depth_range[0]), bar_w, 44, "depth", 
        os.path.join(work, "bar_depth.png"))
    err_bar, err_title = colour_bar(
        work, args.err_cmap, "0 m", "{} m".format(args.max_error), bar_w, 44,
        "absolute error", os.path.join(work, "bar_err.png"))

    legend = os.path.join(work, "legend.png")
    run(["-i", depth_bar, "-i", solid(work, BLOCK_GAP, 44), "-i", err_bar,
         "-filter_complex", "hstack=inputs=3,pad={}:{}:{}:0:white".format(rw, 44, BLOCK_GAP // 2),
         legend])

    titles = os.path.join(work, "titles.png")
    run(["-i", depth_title, "-i", err_title, "-filter_complex",
         "[0]pad={}:{}:0:0:white[a];[1]pad={}:{}:0:0:white[b];[a][b]hstack=inputs=2,"
         "pad={}:{}:{}:0:white".format(bar_w + BLOCK_GAP, 34, bar_w, 34, rw, 34, BLOCK_GAP // 2),
         titles])

    stacked = []
    for i, row in enumerate(grid_rows):
        stacked.append(row)
        stacked.append(solid(work, rw, ROW_GAP))
    stacked += [legend, titles]
    run([a for p in stacked for a in ("-i", p)] +
        ["-filter_complex", "vstack=inputs={}".format(len(stacked)), args.out])
    print("wrote", args.out, size_of(args.out))


if __name__ == "__main__":
    main()
