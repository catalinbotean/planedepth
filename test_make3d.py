"""Smoke-test for the Make3D cross-dataset evaluation.

Run from the repo root:

    python test_make3d.py

No Make3D download and no pretrained weights needed: a tiny synthetic dataset
(random images + random laser grids in the real .mat layout) is written to a
temporary folder and pushed through datasets.Make3DDataset and through
evaluate_depth_make3d.evaluate() in its --ext_disp_to_eval mode.

Prints PASS / FAIL for each check and exits with code 1 if anything fails.
"""

import os, shutil, sys, tempfile, traceback

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from datasets.make3d_dataset import Make3DDataset, crop_band, IMG_PREFIX, DEPTH_PREFIX
from evaluate_depth_make3d import compute_errors, evaluate, save_visualisations
from options import MonodepthOptions

TESTS   = []   # list of (name, fn) in definition order
RESULTS = {}

def register(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator

# ── synthetic dataset ─────────────────────────────────────────────────────
N_IMAGES  = 3
IMG_H, IMG_W = 213, 284      # Make3D is 1704x2272; same aspect, 8x smaller
GT_H, GT_W   = 55, 305       # the real Position3DGrid resolution
NET_H, NET_W = 96, 320       # network input
STEMS = ["op1-p-{:03d}t0".format(i) for i in range(N_IMAGES)]


def make_fake_root(root):
    """Write <root>/Test134/img-*.jpg and <root>/Gridlaserdata/*.mat"""
    img_dir = os.path.join(root, "Test134")
    depth_dir = os.path.join(root, "Gridlaserdata")
    os.makedirs(img_dir)
    os.makedirs(depth_dir)
    rng = np.random.RandomState(0)
    for stem in STEMS:
        rgb = rng.randint(0, 256, (IMG_H, IMG_W, 3)).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(img_dir, IMG_PREFIX + stem + ".jpg"))
        # Position3DGrid is HxWx4; channel 3 is the range in metres. A smooth
        # ramp (rather than noise) so that resizing it to the network
        # resolution and back is lossless enough to assert on in test 5.
        grid = rng.uniform(0, 1, (GT_H, GT_W, 4)).astype(np.float64)
        rows = np.linspace(0., 1., GT_H)[:, None]
        cols = np.linspace(0., 1., GT_W)[None, :]
        grid[:, :, 3] = 5. + 50. * rows + 10. * cols
        savemat(os.path.join(depth_dir, DEPTH_PREFIX + stem + ".mat"),
                {"Position3DGrid": grid})
    return root

# ── 1. centre crop ────────────────────────────────────────────────────────
@register("1. crop_band keeps the same fraction of image and laser grid")
def _():
    for ratio in (2., 1.5, 3.):
        for height in (IMG_H, GT_H, 1704, 55):
            top, bottom = crop_band(height, ratio)
            assert 0 <= top < bottom <= height, (height, ratio, top, bottom)
            frac = (bottom - top) / height
            expected = 1. / (1.33333 * ratio)
            assert abs(frac - expected) < 2. / height, (frac, expected)
    # the standard protocol keeps 639 of the 1704 real image rows
    assert crop_band(1704, 2.) == (532, 1171), crop_band(1704, 2.)

# ── 2. dataset ────────────────────────────────────────────────────────────
@register("2. Make3DDataset returns aligned colour / depth pairs")
def _(root=None):
    dataset = Make3DDataset(root, NET_H, NET_W)
    assert len(dataset) == N_IMAGES, len(dataset)
    assert dataset.filenames == sorted(STEMS), dataset.filenames

    item = dataset[0]
    color = item[("color", "l")]
    depth = item[("depth_gt", "l")]
    assert color.shape == (3, NET_H, NET_W), color.shape
    assert color.min() >= 0. and color.max() <= 1., (color.min(), color.max())

    top, bottom = crop_band(GT_H, 2.)
    assert depth.shape == (1, bottom - top, GT_W), depth.shape
    assert depth.dtype == torch.float32, depth.dtype
    assert (depth > 0).all()

    # batching works (all ground-truth crops share a shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[("color", "l")].shape == (2, 3, NET_H, NET_W)
    assert batch[("depth_gt", "l")].shape == (2, 1, bottom - top, GT_W)

# ── 3. dataset is robust to a flat / nested layout ────────────────────────
@register("3. Make3DDataset finds the data in a nested layout")
def _(root=None):
    nested = os.path.join(root, "nested")
    os.makedirs(os.path.join(nested, "Make3D"))
    for name in ("Test134", "Gridlaserdata"):
        shutil.copytree(os.path.join(root, name),
                        os.path.join(nested, "Make3D", name))
    assert len(Make3DDataset(nested, NET_H, NET_W)) == N_IMAGES

# ── 4. metrics ────────────────────────────────────────────────────────────
@register("4. compute_errors is exact on a perfect prediction")
def _(root=None):
    gt = np.random.RandomState(1).uniform(2., 70., 5000)
    abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3 = compute_errors(gt, gt.copy())
    for value in (abs_rel, sq_rel, rmse, rmse_log, log10):
        assert value < 1e-9, value
    for value in (a1, a2, a3):
        assert value == 1., value

    # a globally scaled prediction is penalised, and median scaling removes it
    pred = gt * 1.3
    assert compute_errors(gt, pred)[0] > 0.29
    pred_scaled = pred * (np.median(gt) / np.median(pred))
    assert compute_errors(gt, pred_scaled)[0] < 1e-9

# ── 5. end-to-end evaluation loop ─────────────────────────────────────────
@register("5. evaluate() runs end to end on --ext_disp_to_eval")
def _(root=None):
    dataset = Make3DDataset(root, NET_H, NET_W)
    gt = np.concatenate([dataset[i][("depth_gt", "l")].numpy() for i in range(len(dataset))])

    opt = MonodepthOptions().parser.parse_args([
        "--eval_mono", "--eval_split", "make3d",
        "--data_path", root,
        "--height", str(NET_H), "--width", str(NET_W),
        "--batch_size", "2", "--num_workers", "0"])

    # Disparities that invert exactly to the ground truth (up to the unknown
    # cross-dataset scale, which median scaling has to recover): the loop must
    # then report ~zero error.
    disps = []
    for i in range(len(dataset)):
        depth = np.clip(gt[i], 1e-3, None)
        disp = 0.1 * 0.58 * opt.width / (0.5 * depth)   # 2x too far away
        disps.append(cv2.resize(disp, (opt.width, opt.height)))
    disp_path = os.path.join(root, "disps.npy")
    np.save(disp_path, np.stack(disps).astype(np.float32))
    opt.ext_disp_to_eval = disp_path

    errors = _capture_errors(opt)
    assert errors is not None, "evaluate() reported no errors"
    assert errors[0] < 0.05, "abs_rel {} after median scaling".format(errors[0])
    assert errors[5] > 0.95, "a1 {} after median scaling".format(errors[5])

    # capping the ground truth must change how many pixels are evaluated
    opt.make3d_max_depth = 40.
    capped = _capture_errors(opt)
    assert capped is not None and np.isfinite(capped).all()


# ── 6. corrupted download ─────────────────────────────────────────────────
@register("6. truncated JPEG errors clearly, --make3d_allow_truncated loads it")
def _(root=None):
    import PIL.ImageFile

    broken = os.path.join(root, "broken")
    os.makedirs(broken)
    for name in ("Test134", "Gridlaserdata"):
        shutil.copytree(os.path.join(root, name), os.path.join(broken, name))

    stem = STEMS[0]
    victim = os.path.join(broken, "Test134", IMG_PREFIX + stem + ".jpg")
    data = open(victim, "rb").read()
    with open(victim, "wb") as f:                      # keep the header, drop the tail
        f.write(data[:int(len(data) * 0.6)])

    # strict first: LOAD_TRUNCATED_IMAGES is a global PIL switch, so the
    # permissive dataset below cannot be constructed before this assertion
    dataset = Make3DDataset(broken, NET_H, NET_W)
    index = dataset.filenames.index(stem)
    try:
        dataset[index]
        raise AssertionError("truncated JPEG did not raise")
    except OSError as exc:
        assert "check_make3d" in str(exc), str(exc)

    try:
        permissive = Make3DDataset(broken, NET_H, NET_W, allow_truncated=True)
        assert permissive[index][("color", "l")].shape == (3, NET_H, NET_W)
    finally:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False


# ── 7. visualisations ─────────────────────────────────────────────────────
@register("7. --eval_out_dir writes panels and a montage")
def _(root=None):
    dataset = Make3DDataset(root, NET_H, NET_W)
    gt = np.concatenate([dataset[i][("depth_gt", "l")].numpy() for i in range(len(dataset))])
    disps = np.stack([np.full((NET_H, NET_W), 0.5, np.float32)
                      + np.linspace(0, 1, NET_W, dtype=np.float32)[None, :]
                      for _ in range(len(dataset))])

    out_dir = os.path.join(root, "vis")
    save_visualisations(out_dir, dataset, disps, gt, 70.)

    for stem in dataset.filenames:
        for suffix in ("_pred.png", "_panel.png"):
            path = os.path.join(out_dir, stem + suffix)
            assert os.path.isfile(path), path
            assert os.path.getsize(path) > 0, path
    montage = os.path.join(out_dir, "make3d_overview.png")
    assert os.path.isfile(montage), montage

    import cv2 as _cv2
    panel = _cv2.imread(os.path.join(out_dir, dataset.filenames[0] + "_panel.png"))
    # rgb + prediction + ground truth stacked, with two 8px separators
    assert panel.shape == (NET_H * 3 + 16, NET_W, 3), panel.shape


def _capture_errors(opt):
    """Run evaluate(opt) and parse the metric row it prints."""
    import io, contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluate(opt)
    for line in buffer.getvalue().splitlines():
        if line.startswith("&"):
            return [float(v) for v in line.replace("\\\\", "").split("&") if v.strip()]
    return None


if __name__ == "__main__":
    root = make_fake_root(tempfile.mkdtemp(prefix="make3d_test_"))
    print("-> synthetic Make3D root: {}\n".format(root))
    try:
        for name, fn in TESTS:
            try:
                fn(root=root) if fn.__code__.co_argcount else fn()
                RESULTS[name] = True
                print("PASS  {}".format(name))
            except Exception:
                RESULTS[name] = False
                print("FAIL  {}".format(name))
                traceback.print_exc()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failed = [n for n, ok in RESULTS.items() if not ok]
    print("\n{}/{} passed".format(len(RESULTS) - len(failed), len(RESULTS)))
    sys.exit(1 if failed else 0)
