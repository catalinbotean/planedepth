# Qualitative figures

The same three steps produce the KITTI and the Make3D comparison figures:
evaluate each model with `--eval_out_dir`, rank the scenes against the ground
truth, then assemble the figure. Every step reads the panel layout from the
files themselves, so the evaluation resolution does not matter.

## 1. Write panels

Each evaluation writes, per test image, `<stem>_panel.png` (input, prediction
and ground truth stacked, separated by 8 px gutters), `<stem>_pred.png` and a
12-scene contact sheet.

**KITTI**, one run per model, into one folder per model:

```shell
python evaluate_depth_scope.py --eval_stereo --eval_split eigen_raw \
--data_path ./kitti --load_weights_folder ./log/ResNet/scope_sd/best_models \
--use_denseaspp --plane_residual --use_mixture_loss --yz_levels 16 \
--use_semantic_gate --use_cross_plane_attn \
--width 1280 --height 384 --eval_out_dir ./qual_kitti/scope_depth
```

For the PlaneDepth baseline use `eval_kitti_original.sh`, which passes only the
flags the published checkpoints were trained with, checks that the ground truth
has been exported and that the frames are `.png`, and writes into
`./qual_kitti/planedepth`:

```shell
bash eval_kitti_original.sh ./log/planedepth_sd/best_models
```

Arguments are optional and positional - `<weights> <width> <height> <kitti_root>
<out_dir>` - and anything after them is forwarded to the evaluation.
`EVAL_SPLIT=eigen_improved` switches the ground truth, `NO_PP=1` drops the flip
post-processing. `--eval_split eigen_improved` gives a much denser
ground truth, which reads far better in a figure than the raw LiDAR.

**Make3D**: the same flag on `evaluate_depth_make3d.py`, and
`scripts/make3d_baseline_eval.py --eval_out_dir` for Monodepth2, Lite-Mono or
TinyDepth. See [MAKE3D.md](MAKE3D.md).

## 2. Rank the scenes

```shell
python scripts/rank_scenes.py ./qual_kitti scope_depth planedepth monodepth2
```

The first folder is the proposed model; the rest are baselines. Scenes where it
does **not** beat every baseline are dropped, the rest are sorted by margin, and
each row ends with the coordinates of the two windows where the gain over the
first baseline is largest. Those coordinates are in an 80x24 grid and go
straight into step 3.

The comparison is structural: the panels are colourised per image, so the maps
are z-scored over the pixels that have a ground-truth measurement and compared
with the z-scored ground truth. It selects scenes and boxes; the numbers in the
paper still come from the metrics table.

## 3. Assemble the figure

```shell
python scripts/build_fig.py ./qual_kitti figures/fig_kitti_qualitative.png \
--scene "2011_09_26_drive_0002_sync_0000000069|(a) urban street with tram tracks|12,12|64,16" \
--scene "2011_09_26_drive_0013_sync_0000000090|(b) cyclist and metal fence|52,16|16,16"
```

Blocks are laid out two per row, each stacking input, every model and the
ground truth at full panel width. Cyan marks the largest-gain window, yellow the
second. Options: `--models`/`--labels` to change the folders and badges (the
proposed model goes last), `--no_gt` to drop the ground-truth row.

## Notes

`scripts/textimg.py` renders the badges and captions by rasterising HTML with
`qlmanage`, because the ffmpeg used here is built without freetype and has no
`drawtext` filter. `scripts/panels.py` holds the shared geometry helpers.
