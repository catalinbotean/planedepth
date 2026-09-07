"""Rank scenes by how much the proposed model beats a baseline, against GT.

Works on any folder tree written by --eval_out_dir (Make3D or KITTI):

    <results>/<ours>/<stem>_panel.png
    <results>/<baseline>/<stem>_panel.png
    ...

The panels are colourised per image, so only structure can be compared, not
absolute depth. Luminance under magma increases with the underlying value, so
each map is z-scored over the pixels where the ground truth has a measurement
(rendered black where it does not) and compared with the z-scored ground truth.
This selects scenes and highlight windows; the quantitative claims stay with
the metrics table.

Usage:
    python scripts/rank_scenes.py <results_dir> [ours] [baseline] [others...]
    python scripts/rank_scenes.py ./qual_kitti scope_depth planedepth monodepth2
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import gray, strips

W, H = 80, 24
BW, BH = 16, 8


def zscore(values, mask):
    kept = [v for v, m in zip(values, mask) if m]
    if len(kept) < 50:
        return None
    mu = sum(kept) / len(kept)
    sd = math.sqrt(sum((v - mu) ** 2 for v in kept) / len(kept)) or 1.0
    return [(v - mu) / sd for v in values]


def load(root, model, stem, index, mask=None, nearest=False):
    path = os.path.join(root, model, stem + "_panel.png")
    crop = strips(path)[index]
    raw = gray(path, crop, W, H, nearest=nearest)
    return raw if mask is None else zscore(raw, mask)


def windows(err_base, err_ours, mask, limit=2):
    """Non-overlapping windows where `ours` beats `base` by the largest margin."""
    scored = []
    for y in range(0, H - BH + 1, 2):
        for x in range(0, W - BW + 1, 4):
            gain = sum(err_base[(y + j) * W + x + i] - err_ours[(y + j) * W + x + i]
                       for j in range(BH) for i in range(BW))
            covered = sum(1 for j in range(BH) for i in range(BW)
                          if mask[(y + j) * W + x + i])
            if covered >= 60:
                scored.append((gain, x, y))
    scored.sort(reverse=True)
    picked = []
    for gain, x, y in scored:
        if gain <= 0:
            break
        if all(abs(x - px) >= BW or abs(y - py) >= BH for _, px, py in picked):
            picked.append((gain, x, y))
        if len(picked) == limit:
            break
    return picked


def main():
    root = sys.argv[1]
    models = sys.argv[2:] or ["scope_depth", "planedepth", "monodepth2"]
    ours, baseline = models[0], models[1]

    stems = sorted(f[:-len("_panel.png")]
                   for f in os.listdir(os.path.join(root, ours))
                   if f.endswith("_panel.png"))

    rows = []
    for stem in stems:
        gt_raw = load(root, ours, stem, 2, nearest=True)
        mask = [v > 3 for v in gt_raw]              # gaps were painted pure black
        if sum(mask) < 200:
            continue
        gt = zscore(gt_raw, mask)
        maps = {m: load(root, m, stem, 1, mask) for m in models}
        if gt is None or any(v is None for v in maps.values()):
            continue

        err = {m: [abs(a - b) if k else 0.0 for a, b, k in zip(maps[m], gt, mask)]
               for m in models}
        n = sum(mask)
        mean = {m: sum(err[m]) / n for m in models}
        if any(mean[ours] >= mean[m] for m in models[1:]):
            continue                                # keep only clean wins
        rows.append((mean[baseline] - mean[ours], mean, stem,
                     windows(err[baseline], err[ours], mask)))

    rows.sort(reverse=True)
    print("{} scenes where {} beats every baseline\n".format(len(rows), ours))
    header = "{:>6}  " + "  ".join("{:>10}" for _ in models) + "  {:<26} boxes"
    print(header.format("gain", *models, "scene"))
    for gain, mean, stem, boxes in rows[:20]:
        cells = ["{:10.3f}".format(mean[m]) for m in models]
        boxtxt = " ".join("{},{}".format(x, y) for _, x, y in boxes)
        print("{:6.3f}  {}  {:<26} {}".format(gain, "  ".join(cells), stem, boxtxt))


if __name__ == "__main__":
    main()
