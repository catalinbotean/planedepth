"""Make3D test-set loader, used for cross-dataset generalisation evaluation.

The Make3D test set ("Test134") ships 134 outdoor images at 2272x1704 together
with laser range maps stored as MATLAB files on a 55x305 spherical grid
(``Position3DGrid[:, :, 3]`` is the range in metres).

Because the Make3D images are almost square while the model is trained on the
wide KITTI aspect ratio, the standard evaluation protocol (Godard et al.,
FAL-Net, PLADE-Net) keeps only a horizontal band of the image centred on the
principal point before resizing to the network resolution::

    h_ratio = 1 / (1.33333 * ratio)     # ratio = 2 by default
    image   = image[(1 - h_ratio) * H / 2 : (1 + h_ratio) * H / 2, :]

The ground-truth grid covers the same field of view, so exactly the same
fractional band is cropped from the 55-row depth map, which keeps predictions
and ground truth aligned.

Download (both archives extract into the same root)::

    http://make3d.cs.cornell.edu/data/Test134.tar.gz
    http://make3d.cs.cornell.edu/data/Gridlaserdata.tar.gz

giving::

    <data_path>/Test134/img-<stem>.jpg
    <data_path>/Gridlaserdata/depth_sph_corr-<stem>.mat
"""

from __future__ import absolute_import, division, print_function

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision

from .mono_dataset import pil_loader

IMG_PREFIX = "img-"
DEPTH_PREFIX = "depth_sph_corr-"


def crop_band(height, crop_ratio):
    """Rows kept by the standard Make3D centre crop, as (top, bottom)."""
    h_ratio = 1. / (1.33333 * crop_ratio)
    top = int(height * (1. - h_ratio) / 2.)
    bottom = int(height * (1. + h_ratio) / 2.)
    return top, bottom


class Make3DDataset(data.Dataset):
    """Make3D Test134 dataset.

    Returns the same keys the KITTI loaders produce at test time, so the same
    evaluation loop can be used:

        ("color", "l")   B, 3, height, width      network input
        ("depth_gt", "l") B, 1, gt_h, gt_w        cropped laser range map (m)
        "index"                                   index into ``self.filenames``
    """

    def __init__(self,
                 data_path,
                 height,
                 width,
                 filenames=None,
                 crop_ratio=2.,
                 img_dir="Test134",
                 depth_dir="Gridlaserdata",
                 img_ext=".jpg"):
        super(Make3DDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.crop_ratio = crop_ratio
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = torchvision.transforms.ToTensor()

        self.img_path = self._resolve_dir(img_dir, IMG_PREFIX + "*" + img_ext)
        self.depth_path = self._resolve_dir(depth_dir, DEPTH_PREFIX + "*.mat")

        if filenames is None:
            filenames = self._scan_stems()
        self.filenames = filenames

        assert len(self.filenames) > 0, \
            "No Make3D test images found under {}".format(self.data_path)

    def _resolve_dir(self, name, pattern):
        """Find ``name`` under data_path, falling back to data_path itself.

        Different Make3D mirrors nest the archives differently (some extract
        directly into the root), so we accept any of those layouts rather than
        failing on a path that does contain the data.
        """
        candidates = [os.path.join(self.data_path, name), self.data_path]
        candidates += sorted(glob.glob(os.path.join(self.data_path, "*", name)))
        for candidate in candidates:
            if glob.glob(os.path.join(candidate, pattern)):
                return candidate
        raise FileNotFoundError(
            "Could not find any '{}' under {} (looked for a '{}' folder). "
            "Point --data_path at the folder holding the extracted Test134 "
            "and Gridlaserdata archives.".format(pattern, self.data_path, name))

    def _scan_stems(self):
        """All image stems that also have a ground-truth range map, sorted."""
        stems = []
        missing = 0
        pattern = os.path.join(self.img_path, IMG_PREFIX + "*" + self.img_ext)
        for image in sorted(glob.glob(pattern)):
            stem = os.path.basename(image)[len(IMG_PREFIX):-len(self.img_ext)]
            if os.path.isfile(self.get_depth_path(stem)):
                stems.append(stem)
            else:
                missing += 1
        if missing:
            print("-> Make3D: skipping {} image(s) without a matching {}*.mat"
                  .format(missing, DEPTH_PREFIX))
        return stems

    def get_image_path(self, stem):
        return os.path.join(self.img_path, IMG_PREFIX + stem + self.img_ext)

    def get_depth_path(self, stem):
        return os.path.join(self.depth_path, DEPTH_PREFIX + stem + ".mat")

    def get_color(self, stem):
        """Centre-cropped image resized to the network resolution."""
        color = self.to_tensor(self.loader(self.get_image_path(stem)))
        top, bottom = crop_band(color.shape[1], self.crop_ratio)
        color = color[:, top:bottom, :]
        # Same resampling as datasets.pair_transforms.Resize, so the test-time
        # preprocessing matches the one used during training.
        color = F.interpolate(color[None, ...], size=(self.height, self.width),
                              mode="bicubic", align_corners=True)[0]
        return color.clamp(min=0., max=1.)

    def get_depth(self, stem):
        """Centre-cropped laser range map, in metres."""
        from scipy.io import loadmat  # imported lazily: only Make3D needs scipy

        depth_gt = loadmat(self.get_depth_path(stem))["Position3DGrid"][:, :, 3]
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        top, bottom = crop_band(depth_gt.shape[0], self.crop_ratio)
        return depth_gt[top:bottom, :]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        stem = self.filenames[index]
        inputs = {}
        inputs[("color", "l")] = self.get_color(stem)
        inputs[("depth_gt", "l")] = torch.from_numpy(
            np.expand_dims(self.get_depth(stem), 0))
        inputs["index"] = index
        return inputs
