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

Roughly 400 MB. Take the test-set images and the matching laser range grid
from the official data page — <http://make3d.cs.cornell.edu/data.html> — and
extract both archives into the same folder. The page is HTTP only, and the
direct archive URLs have changed more than once, so copy them from there rather
than from a hard-coded link.

The result must look like:

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

## Rularea checkpoint-urilor originale PlaneDepth

Modelele publicate cu articolul (`stage1`, `HRfinetune`, `self-distillation` —
link-urile din tabelul "Pretrained model" al `README.md`) se evalueaza cu
`eval_make3d_original.sh`, care trimite doar flag-urile cu care au fost
antrenate, fara niciuna dintre extensiile adaugate ulterior:

```shell
bash eval_make3d_original.sh
```

Fara argumente cauta singur folderul care contine `depth.pth`, foloseste
1280x384 si `./make3d`. Argumentele sunt optionale, in ordinea asta:

```shell
bash eval_make3d_original.sh <weights_folder> [width] [height] [make3d_root] [flag-uri in plus]
```

Orice pui dupa al patrulea argument se trimite mai departe la
`evaluate_depth_make3d.py` (de exemplu `--num_workers 4` sau
`--make3d_max_depth 80`).

Scriptul trimite implicit `--make3d_allow_truncated`, pentru ca o imagine din
Test134 vine trunchiata cu 31 de octeti din arhiva originala, iar randurile
lipsa cad in afara benzii centrale pastrate de protocol, deci nu schimba nicio
metrica. Cu `MAKE3D_STRICT=1` in fata comenzii, flag-ul nu mai este trimis.

| Checkpoint | width | height |
|------------|-------|--------|
| `stage1` | 640 | 192 |
| `HRfinetune` | 1280 | 384 |
| `self-distillation` | 1280 | 384 |

Exemplu:

```shell
bash eval_make3d_original.sh ./log/planedepth_sd/best_models 1280 384 ./make3d
```

Arhiva descarcata trebuie sa contina `encoder.pth` si `depth.pth` direct in
folderul dat ca `<weights_folder>`.

Modulele adaugate dupa publicare (`--use_semantic_gate`,
`--use_cross_plane_attn`, `--num_learned_families`, `--use_multiscale_logits`,
`--adaptive_plane_range`, `--pixelwise_plane_residual`) isi creeaza parametrii
doar cand flag-ul lor este activ, deci cu flag-urile de mai sus `state_dict`-ul
original se incarca strict, fara chei lipsa. Daca apare
`Missing key(s) in state_dict`, inseamna ca a ramas activat un flag in plus
fata de cele cu care a fost antrenat checkpoint-ul.

## Imagini rezultate

Cu `--eval_out_dir` scriptul salveaza, dupa metrici, cate un panou pentru
fiecare imagine plus un contact sheet:

```shell
bash eval_make3d_original.sh ./log/planedepth_sd/best_models 1280 384 ./make3d --eval_out_dir ./qual_make3d
```

In folder apar:

| Fisier | Continut |
|--------|----------|
| `<stem>_panel.png` | imaginea de intrare, predictia si ground truth-ul, unul sub altul |
| `<stem>_pred.png` | doar predictia, colorata cu magma (aproape = deschis) |
| `make3d_overview.png` | 12 scene alese uniform, intr-o grila 4x3, pentru o privire rapida |

Ground truth-ul este grila laser de 20x305 randuri pastrate, afisata cu
nearest-neighbour ca sa se vada rezolutia ei reala; pixelii fara masuratoare
si cei peste `--make3d_max_depth` raman negri. Predictia este disparitate, deci
scara de culoare nu este metrica — pentru adancimi in metri foloseste
`--save_pred_disps` si aplici tu scalarea.

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

## Verificarea arhivei descarcate

Un JPEG incomplet opreste evaluarea in mijlocul ei, cu
`OSError: image file is truncated`, si omoara un worker de dataloader. Inainte
de prima rulare:

```shell
python scripts/check_make3d.py ./make3d
```

Verifica faptul ca fiecare imagine se decodeaza complet, ca fiecare imagine are
un `.mat` pereche si ca fiecare `.mat` contine un `Position3DGrid` citibil.
Iese cu 0 daca totul e in regula, altfel listeaza fisierele problema.

Daca un fisier ramane trunchiat si dupa un download proaspat, este unul dintre
JPEG-urile scurte din arhiva originala. Atunci se poate rula cu:

```shell
--make3d_allow_truncated
```

Randurile lipsa se decodeaza gri, deci metricile imaginii respective sunt
usor deplasate — cu 1-2 imagini din 134 efectul asupra mediei e mic, dar real.
Fara acest flag, loader-ul se opreste cu un mesaj care spune exact ce fisier
este si ce optiuni ai.

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
| `scripts/check_make3d.py` | Verifica integritatea datasetului descarcat |
| `eval_make3d_original.sh` | Run command for the released PlaneDepth checkpoints |
