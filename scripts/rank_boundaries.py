"""Rank scenes by how well a model's depth discontinuities sit on image edges.

Without ground truth nothing can be scored for accuracy, but boundary
behaviour can: a good depth map concentrates its gradient where the image has
a real edge and stays flat elsewhere. The score is

    mean |grad depth| over the strongest image edges
    ------------------------------------------------
    mean |grad depth| over the flattest half of the image

so it rewards crisp object contours and penalises depth that ripples across
textureless surfaces. It is a selection heuristic, not a metric to report.

Usage:
    python scripts/rank_boundaries.py <results_dir> <ours> <baseline> [more...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panels import gray, strips

W, H = 320, 96
BW, BH = 16, 8
GRID_W, GRID_H = 80, 24


def gradient(values, w, h):
    """Forward-difference gradient magnitude, as a flat list."""
    out = [0.0] * (w * h)
    for y in range(h - 1):
        row = y * w
        for x in range(w - 1):
            i = row + x
            dx = values[i + 1] - values[i]
            dy = values[i + w] - values[i]
            out[i] = abs(dx) + abs(dy)
    return out


def boundary_score(image_grad, depth_grad, edge_idx, flat_idx):
    edge = sum(depth_grad[i] for i in edge_idx) / max(len(edge_idx), 1)
    flat = sum(depth_grad[i] for i in flat_idx) / max(len(flat_idx), 1)
    return edge / (flat + 1e-6)


def main():
    root = sys.argv[1]
    models = sys.argv[2:]
    if len(models) < 2:
        raise SystemExit("give the proposed model and at least one baseline")
    ours = models[0]

    stems = sorted(f[:-len("_panel.png")]
                   for f in os.listdir(os.path.join(root, ours))
                   if f.endswith("_panel.png"))
    stems = [s for s in stems
             if all(os.path.isfile(os.path.join(root, m, s + "_panel.png"))
                    for m in models)]

    rows = []
    for n, stem in enumerate(stems, start=1):
        panel = os.path.join(root, ours, stem + "_panel.png")
        crops = strips(panel)
        image = gray(panel, crops[0], W, H)
        ig = gradient(image, W, H)

        order = sorted(range(W * H), key=lambda i: ig[i])
        flat_idx = order[:len(order) // 2]
        edge_idx = order[-len(order) // 10:]

        scores = {}
        for model in models:
            p = os.path.join(root, model, stem + "_panel.png")
            depth = gray(p, strips(p)[1], W, H)
            scores[model] = boundary_score(ig, gradient(depth, W, H),
                                           edge_idx, flat_idx)

        best_baseline = max(scores[m] for m in models[1:])
        rows.append((scores[ours] - best_baseline, scores, stem, edge_idx))
        if n % 200 == 0:
            print("   {}/{}".format(n, len(stems)), file=sys.stderr, flush=True)

    rows.sort(reverse=True)
    print("{} scenes, sorted by boundary sharpness of {} minus the best baseline\n"
          .format(len(rows), ours))
    print(("{:>7}  " + "  ".join("{:>11}" for _ in models) + "  {:<40} boxes")
          .format("margin", *models, "scene"))

    for margin, scores, stem, edge_idx in rows[:25]:
        # the two windows holding the most image edge, where the difference shows
        counts = {}
        for i in edge_idx:
            gx = (i % W) * GRID_W // W
            gy = (i // W) * GRID_H // H
            counts[(gx // 4 * 4, gy // 2 * 2)] = counts.get((gx // 4 * 4, gy // 2 * 2), 0) + 1
        best = sorted(counts.items(), key=lambda kv: -kv[1])
        picked = []
        for (x, y), _ in best:
            x = min(x, GRID_W - BW)
            y = min(y, GRID_H - BH)
            if all(abs(x - px) >= BW or abs(y - py) >= BH for px, py in picked):
                picked.append((x, y))
            if len(picked) == 2:
                break
        cells = "  ".join("{:11.3f}".format(scores[m]) for m in models)
        print("{:7.3f}  {}  {:<40} {}".format(
            margin, cells, stem, " ".join("{},{}".format(x, y) for x, y in picked)))


if __name__ == "__main__":
    main()
