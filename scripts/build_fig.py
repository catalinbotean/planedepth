"""Assemble a qualitative figure from panels written by --eval_out_dir.

Each scene becomes a block that stacks input / baselines / ours / ground truth
at full panel width, so the detail a reviewer is asked to look at stays
legible, and the blocks are laid out in a two-column grid. Highlight boxes are
given in the 80x24 grid that scripts/rank_scenes.py reports.

Works for any dataset and evaluation resolution: the strip geometry is read
from the panels themselves.

Usage:
    python scripts/build_fig.py <results_dir> <out.png> \
        --scene "gatesback4-p-139t0|(a) hedge in front of parked bicycles|12,12|64,16" \
        --scene "op20-p-046t000|(b) ornamented facade with a flower bed|52,16|16,16"

    --models  comma-separated folders, ours LAST (default:
              monodepth2,planedepth,scope_depth)
    --labels  comma-separated badges for those folders
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import size_of, strip_count, strips
from textimg import render

GUT, BLOCK_GAP, ROW_GAP = 6, 44, 60
CYAN, YELLOW = "#22d3ee", "#facc15"
GRID_W, GRID_H = 80, 24


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y"] + args, check=True)


def solid(work, w, h):
    """A white spacer of the given size, cached by size so runs cannot collide."""
    path = os.path.join(work, "spacer_{}x{}.png".format(w, h))
    if not os.path.isfile(path):
        run(["-f", "lavfi", "-i", "color=white:s={}x{}".format(w, h), "-frames:v", "1", path])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--scene", action="append", required=True,
                    help="stem|caption|x1,y1[|x2,y2]")
    ap.add_argument("--models", default="monodepth2,planedepth,scope_depth")
    ap.add_argument("--labels", default="Monodepth2,PlaneDepth,SCOPE-Depth (ours)")
    ap.add_argument("--no_gt", action="store_true", help="omit the ground-truth row")
    ap.add_argument("--gt_dir",
                    help="folder of <stem>_gtcolor.png images to append as a "
                         "ground-truth row, for panels that carry none "
                         "(see scripts/fetch_kitti_gt.py and colorize_gt.py)")
    args = ap.parse_args()

    models = args.models.split(",")
    labels = args.labels.split(",")
    assert len(models) == len(labels), "--models and --labels must match"

    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_fig_work")
    os.makedirs(work, exist_ok=True)

    scenes = []
    for spec in args.scene:
        parts = spec.split("|")
        stem, caption = parts[0], parts[1]
        boxes = [tuple(int(v) for v in p.split(",")) for p in parts[2:] if p]
        scenes.append((stem, caption, boxes))

    # every panel is resampled to the geometry of the last model's panels
    ref = os.path.join(args.results, models[-1], scenes[0][0] + "_panel.png")
    pw, ph = size_of(ref)
    ph = (ph - 16) // 3

    # rows of a block: input, each model, and the ground truth
    rows = [("Input", models[-1], 0)]
    rows += [(labels[i], models[i], 1) for i in range(len(models))]
    if not args.no_gt and strip_count(ref) == 3:
        rows.append(("Ground truth", models[-1], 2))
    elif args.gt_dir:
        rows.append(("Ground truth", args.gt_dir, None))

    badges = {}
    for label, _, _ in rows:
        key = "".join(c for c in label if c.isalnum())
        badges[label] = os.path.join(work, "badge_" + key + ".png")
        if not os.path.isfile(badges[label]):
            render("<span style=\"display:inline-block;background:#151515;color:#fff;"
                   "font-family:Helvetica,Arial,sans-serif;font-size:13px;"
                   "font-weight:600;padding:3px 7px\">{}</span>".format(label),
                   badges[label], max(28, ph // 8))

    blocks = []
    for stem, caption, boxes in scenes:
        safe = stem.replace(".", "_").replace("/", "_")
        cap_raw = os.path.join(work, safe + "_capraw.png")
        render("<span style=\"font-family:Palatino,Georgia,serif;font-size:14px;"
               "color:#111\">{}</span>".format(caption), cap_raw, max(26, ph // 9))
        cap = os.path.join(work, safe + "_cap.png")
        run(["-i", cap_raw, "-vf", "pad={}:{}:0:0:white".format(pw, ph // 7 + 12), cap])

        draw = ""
        for (gx, gy), colour in zip(boxes, (CYAN, YELLOW)):
            draw += ",drawbox=x={}:y={}:w={}:h={}:color={}@1.0:t=7".format(
                gx * pw // GRID_W, gy * ph // GRID_H,
                16 * pw // GRID_W, 8 * ph // GRID_H, colour)

        parts = [cap]
        for i, (label, model, strip) in enumerate(rows):
            out = os.path.join(work, "{}_{}.png".format(safe, i))
            if strip is None:                     # a ready-made image, not a panel
                src = os.path.join(model, stem + "_gtcolor.png")
                if not os.path.isfile(src):
                    raise SystemExit("no ground truth for {} in {}".format(stem, model))
                crop = "{}:{}:0:0".format(*size_of(src))
            else:
                src = os.path.join(args.results, model, stem + "_panel.png")
                crop = strips(src)[strip]
            run(["-i", src, "-i", badges[label], "-filter_complex",
                 "[0]crop={},scale={}:{}{}[p];[p][1]overlay=18:16".format(
                     crop, pw, ph, draw), out])
            parts.append(out)
            if i < len(rows) - 1:
                parts.append(solid(work, pw, GUT))

        block = os.path.join(work, "block_" + safe + ".png")
        run([a for p in parts for a in ("-i", p)] +
            ["-filter_complex", "vstack=inputs={}".format(len(parts)), block])
        blocks.append(block)

    if len(blocks) == 1:
        run(["-i", blocks[0], "-c", "copy", args.out])
        print("wrote", args.out, size_of(args.out))
        return

    bw, bh = size_of(blocks[0])
    vgap = solid(work, BLOCK_GAP, bh)
    grid_rows = []
    for i in range(0, len(blocks), 2):
        pair = blocks[i:i + 2]
        row = os.path.join(work, "row{}.png".format(i))
        if len(pair) == 2:
            run(["-i", pair[0], "-i", vgap, "-i", pair[1],
                 "-filter_complex", "hstack=inputs=3", row])
        else:
            run(["-i", pair[0], "-vf",
                 "pad={}:{}:0:0:white".format(2 * bw + BLOCK_GAP, bh), row])
        grid_rows.append(row)

    rw, _ = size_of(grid_rows[0])
    hgap = solid(work, rw, ROW_GAP)
    stacked = []
    for i, row in enumerate(grid_rows):
        stacked.append(row)
        if i < len(grid_rows) - 1:
            stacked.append(hgap)
    run([a for p in stacked for a in ("-i", p)] +
        ["-filter_complex", "vstack=inputs={}".format(len(stacked)), args.out])
    print("wrote", args.out, size_of(args.out))


if __name__ == "__main__":
    main()
