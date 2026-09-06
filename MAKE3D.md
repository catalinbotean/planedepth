# Make3D evaluation

Cross-dataset generalisation test: a KITTI-trained PlaneDepth model is evaluated
on the 134 Make3D test images without any fine-tuning.

Everything below is run from the repo root.

## 1. Install scipy

The only new dependency — it reads the `.mat` ground-truth range maps.

```shell
pip install "scipy>=1.10"
```

(It is also in `requirements.txt`, so `pip install -r requirements.txt` covers it.)

## 2. Smoke-test the code

Runs the loader and the full evaluation loop on a synthetic dataset. No download
and no pretrained weights needed, takes a few seconds.

```shell
python test_make3d.py
```

## 3. Download the Make3D test set

Roughly 400 MB, the two archives extract into the same folder.

```shell
mkdir -p make3d && cd make3d
wget http://make3d.cs.cornell.edu/data/Test134.tar.gz http://make3d.cs.cornell.edu/data/Gridlaserdata.tar.gz
tar -xzf Test134.tar.gz && tar -xzf Gridlaserdata.tar.gz
cd ..
```

This gives:

```
make3d/Test134/img-<stem>.jpg
make3d/Gridlaserdata/depth_sph_corr-<stem>.mat
```

The loader also accepts these two folders nested one level deeper (e.g.
`make3d/Make3D/Test134/`), and skips any image without a matching `.mat`.

## 4. Run the evaluation

Edit `--load_weights_folder` and `--data_path` in `eval_make3d.sh`, then:

```shell
bash eval_make3d.sh
```

Or directly:

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth_make3d.py \
--eval_stereo \
--eval_split make3d \
--data_path ./make3d \
--load_weights_folder ./log/ResNet/exp1_sd/best_models \
--use_denseaspp \
--plane_residual \
--use_mixture_loss \
--post_process \
--batch_size 1 \
--width 1280 \
--height 384
```

**The architecture flags must match how the model was trained**, exactly as with
`eval.sh`. If the checkpoint was trained with `--use_semantic_gate`,
`--use_cross_plane_attn`, `--num_learned_families`, `--use_multiscale_logits`,
`--adaptive_plane_range`, `--pixelwise_plane_residual` … pass the same flags
here or loading the `state_dict` will fail. `--width/--height` must be the
resolution the checkpoint was trained or finetuned at (384x1280 for the
HR-finetuned models, 192x640 for stage 1).

Output is a single metric row:

```
   abs_rel |   sq_rel |     rmse | rmse_log |    log10 |       a1 |       a2 |       a3 |
```

## Protocol

Implemented in `evaluate_depth_make3d.py` and `datasets/make3d_dataset.py`,
following the standard Make3D protocol used by Godard et al., FAL-Net and
PLADE-Net:

* **Centre crop.** Make3D images are nearly square while the model is trained on
  the wide KITTI aspect ratio, so only a horizontal band of
  `1 / (1.33333 * ratio)` of the height is kept, centred on the principal point
  (`--make3d_crop_ratio`, default 2, i.e. 639 of the 1704 rows). The 55x305
  laser grid covers the same field of view and is cropped by the same fraction,
  which keeps prediction and ground truth aligned.
* **Scale.** Make3D shares neither the baseline nor the focal length of the
  KITTI rig, so the prediction is only defined up to scale and it is recovered
  per image by median scaling. The stereo factor 5.4 does not transfer; passing
  `--eval_stereo --disable_median_scaling` prints a warning saying so.
* **Depth cap.** Errors are accumulated over ground-truth depths below
  `--make3d_max_depth`, 70 m by default — the "C1" error of the Make3D
  literature. Pass `--make3d_max_depth 80` for C2.
* **log10** is reported in addition to the usual KITTI metrics, since that is
  what the Make3D tables use.

## Other options

| Flag | Meaning |
|------|---------|
| `--post_process` | Monodepth flip post-processing (two forward passes per image) |
| `--save_pred_disps` | Saves `disps_make3d_split.npy` into the weights folder |
| `--ext_disp_to_eval <file.npy>` | Scores previously saved disparities instead of running the network |
| `--no_eval` | Predict only, skip the metrics |
| `--batch_size`, `--num_workers` | Standard loader settings |

Predictions are ordered by the sorted image stems. To pin a subset or a specific
ordering, drop a `splits/make3d/test_files.txt` with one stem per line (e.g.
`op1-p-108t0`) — the script uses it automatically when present, and otherwise
scans the directory.

## Files

| File | Role |
|------|------|
| `evaluate_depth_make3d.py` | Evaluation entry point |
| `datasets/make3d_dataset.py` | `Make3DDataset` — Test134 loader, centre crop, `.mat` ground truth |
| `eval_make3d.sh` | Default run command |
| `test_make3d.py` | Smoke test on a synthetic dataset |
