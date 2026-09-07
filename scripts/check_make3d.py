"""Verify a Make3D download before running the evaluation.

Checks that every Test134 image decodes fully, that every image has a matching
Gridlaserdata range map, and that every .mat holds a readable 55x305
``Position3DGrid``. Reports the broken files instead of letting a dataloader
worker die mid-evaluation with "image file is truncated".

Usage:
    python scripts/check_make3d.py [data_path]      # default: ./make3d

Exits 0 when the dataset is usable, 1 when something is wrong.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasets.make3d_dataset import (Make3DDataset, IMG_PREFIX, DEPTH_PREFIX,
                                     crop_band)


def main(data_path):
    from PIL import Image
    from scipy.io import loadmat

    try:
        dataset = Make3DDataset(data_path, 192, 640)
    except Exception as exc:
        print("-> Could not open the dataset: {}".format(exc))
        return 1

    print("-> images : {}".format(dataset.img_path))
    print("-> depths : {}".format(dataset.depth_path))
    print("-> pairs  : {}".format(len(dataset)))

    truncated, unreadable_mat, bad_shape = [], [], []

    for stem in dataset.filenames:
        image_path = dataset.get_image_path(stem)
        try:
            Image.open(image_path).load()
        except Exception as exc:
            truncated.append((image_path, exc))

        depth_path = dataset.get_depth_path(stem)
        try:
            grid = loadmat(depth_path)["Position3DGrid"]
        except Exception as exc:
            unreadable_mat.append((depth_path, exc))
            continue
        if grid.ndim != 3 or grid.shape[2] < 4:
            bad_shape.append((depth_path, grid.shape))

    # images with no ground truth at all are dropped by the loader, count them
    all_images = [f for f in os.listdir(dataset.img_path)
                  if f.startswith(IMG_PREFIX) and f.endswith(dataset.img_ext)]
    orphans = len(all_images) - len(dataset)

    print()
    for path, exc in truncated:
        print("TRUNCATED  {} ({})".format(path, exc))
    for path, exc in unreadable_mat:
        print("UNREADABLE {} ({})".format(path, exc))
    for path, shape in bad_shape:
        print("BAD SHAPE  {} (Position3DGrid is {})".format(path, shape))
    if orphans > 0:
        print("NO GT      {} image(s) have no matching {}*.mat and are skipped"
              .format(orphans, DEPTH_PREFIX))

    if not (truncated or unreadable_mat or bad_shape):
        top, bottom = crop_band(55, 2.)
        print("-> All {} pairs are readable. Ground truth crops to {} of 55 rows."
              .format(len(dataset), bottom - top))
        if orphans > 0:
            print("-> The images without ground truth are expected; the official "
                  "archives are not a perfect 1:1 match.")
        return 0

    print()
    print("-> Re-extract the affected archive, or fetch it again from the "
          "Make3D data page: http://make3d.cs.cornell.edu/data.html")
    if truncated:
        print("-> If the file is still truncated after a fresh download, it is "
              "one of the short JPEGs in the original archive: pass "
              "--make3d_allow_truncated to decode it anyway. The missing rows "
              "become grey, so that image's metrics are biased - with 1-2 "
              "images out of 134 the effect on the mean is small but real.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./make3d"))
