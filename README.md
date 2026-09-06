# PlaneDepth

This is the official PyTorch implementation for the CVPR 2023 paper

> **PlaneDepth: Self-supervised Depth Estimation via Orthogonal Planes**
> 
> [Arxiv](https://arxiv.org/abs/2210.01612)

<p align="center">
  <img src="figures/pipeline.png" alt="pipeline of our method" width="1000" />
</p>

## 🐁 Setup
We recommend using anaconda to create the env and install the requirements by running:
```shell
conda create -n planedepth python=3.9
conda activate planedepth
# CUDA 12.1 wheels — required for sm_90 GPUs (Hopper / H100). The cu121 build
# ships kernels for sm_70/75/80/86/89/90, so it also runs on older GPUs.
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
> **Older GPUs (pre-Hopper):** the cu121 build above works everywhere from
> Volta (sm_70) up. If you specifically need CUDA 11.8 instead, use
> `--index-url https://download.pytorch.org/whl/cu118` (also includes sm_90).

## 🐂 KITTI training data
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti/
```
Then unzip with
```shell
cd kitti
unzip "*.zip"
cd ..
```
You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

## 🐅 Training
We provide the defult stereo training command of stage1 in `train_ResNet.sh`.

To perform HRfinetune after stage1, update `train_ResNet.sh` as:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train.py \
--png \
--model_name exp1_HR \ # modified
--use_denseaspp \
--use_mixture_loss \
--plane_residual \
--flip_right \
--learning_rate 2.5e-5 \ # modified
--num_epochs 1 \ # modified
--width 1280 \ # new
--height 384 \ # new
--no_crop \ # new
--load_weights_folder ./log/ResNet/exp1/last_models \ # new
--models_to_load depth encoder semantic_gate # new (semantic_gate only if stage1 used --use_semantic_gate)
```
> **Carrying the semantic gate forward:** if stage 1 was trained with
> `--use_semantic_gate`, you **must** add `semantic_gate` to `--models_to_load`
> *and* keep `--use_semantic_gate` set here, otherwise the learned gate is
> re-initialised to zero and the semantic prior is silently dropped at exactly
> the high-res stage where it helps most. Optionally enable
> `--semantic_edge_smoothness` here (semantic-boundary-aware smoothness).

To perform self-distillation after HRfinetune, update `train_ResNet.sh` as:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train.py \
--png \
--model_name exp1_sd \ # modified
--use_denseaspp \
--use_mixture_loss \
--plane_residual \
--batch_size 4 \ # modified
--learning_rate 2e-5 \ # modified
--num_epochs 10 \ # modified
--milestones 5 \ # modified
--width 1280 \
--height 384 \
--no_crop \
--load_weights_folder ./log/ResNet/exp1_HR/last_models \ # modified
--models_to_load depth encoder semantic_gate \ # add semantic_gate if using the gate
--self_distillation 1. # new
```
> **Semantic-guided self-distillation (optional, paper experiments):** with
> `--use_semantic_gate` active, add `--semantic_distill_weight 1.0` to trust the
> teacher's pseudo-labels more where the segmenter's expected plane family
> (road→XZ, wall→YZ, sky→XY) agrees with the model's own family assignment, and
> down-weight ambiguous pixels. Combine with
> `--uncertainty_weighted_distillation` for confidence × semantic weighting.

**Monocular training:**

Please adjust the following flags:
```shell
--warp_type homography_warp
--split eigen_zhou
--novel_frame_ids 1 -1
--automask
(optional) --no_stereo
(optional) --use_colmap
```

**Other training options**

Look at `options.py` to see other options.


## 🐇 Evaluation

We provide the defult evaluation command in `eval.sh`. Please refer to your training settings to modify it.

**Prepare Eigen raw ground truth**

You may need to export the ground truth depth before evaluation on Eigen raw split. Please run:
```shell
python splits/eigen_raw/export_gt_depth.py --data_path ./kitti
```

**Prepare Eigen improved ground truth**

To perform Eigen improved evaluation, you need to download the [Eigen improved dataset (14GB)](https://www.cvlibs.net/datasets/kitti/eval_depth.php) and unzip it by running:
```shell
unzip data_depth_annotated.zip -d kitti_depth
```

You can also place it wherever you like and point towards it with the --improved_path flag during export:
```shell
python splits/eigen_improved/prepare_groundtruth.py --improved_path ./kitti_depth
```

**Make3D (cross-dataset generalisation)**

`evaluate_depth_make3d.py` tests a KITTI-trained model on the 134 Make3D test
images without any fine-tuning. Download and extract the two archives into one
folder:
```shell
mkdir make3d && cd make3d
wget http://make3d.cs.cornell.edu/data/Test134.tar.gz http://make3d.cs.cornell.edu/data/Gridlaserdata.tar.gz
tar -xzf Test134.tar.gz && tar -xzf Gridlaserdata.tar.gz && cd ..
```
giving `make3d/Test134/img-*.jpg` and `make3d/Gridlaserdata/depth_sph_corr-*.mat`
(the loader also accepts these two folders nested one level deeper). Then run
`eval_make3d.sh`, pointing `--data_path` at that folder and using the same
architecture flags the model was trained with:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth_make3d.py \
--eval_stereo --eval_split make3d --data_path ./make3d \
--load_weights_folder ./log/ResNet/exp1_sd/best_models \
--use_denseaspp --plane_residual --use_mixture_loss \
--post_process --batch_size 1 --width 1280 --height 384
```
Following the standard protocol, image and 55x305 laser grid are both centre
cropped to the same horizontal band (`--make3d_crop_ratio`, default 2), the
per-image scale is recovered by median scaling (Make3D shares neither the
baseline nor the focal length of the KITTI rig), and errors are accumulated
over ground-truth depths below `--make3d_max_depth` (70m, the C1 protocol).
The script reports `log10` in addition to the usual metrics.

`python test_make3d.py` smoke-tests the loader and the evaluation loop on a
synthetic dataset — no download and no weights required.

See [MAKE3D.md](MAKE3D.md) for the full step-by-step instructions, the protocol
details and the remaining options.

## 🐉 Pretrained model

| Model      | Abs Rel |  A1  |
|------------|------|-------------|
| [`stage1`](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wangry3_shanghaitech_edu_cn/EiLFuTdtmidMgu-1XbNpr9wBiKk4NZbNv60RfxajlfDiWA?e=cmfpwc) | 0.089                | 0.900        |
| [`HRfinetune`](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wangry3_shanghaitech_edu_cn/EqmusgpF_m5GmwpmsG7czO4ByfFIIJe450GsFvST9mUn_w?e=grQJFz) | 0.086                | 0.906      |
| [`self-distillation`](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wangry3_shanghaitech_edu_cn/EmYCmInpVd5CjJwu8-DCyY4BnJTKQ7IKnRx5GJYqQEVeMg?e=OCRdEl) | 0.085             | 0.910       |

## 🐍 Known issues
- When using the flag --use_mixture_loss in the `train.py`, users may encounter the error message "CUDNN_STATUS_NOT_INITIALIZED". This issue may be resolved by reducing the batch_size. [Issue_4](https://github.com/svip-lab/PlaneDepth/issues/4)

## 🐎 TODO list

- [x] The ground truth depth during training is wrong because of cropping, which will influence the training log in tensorboard.

## 🐐 Acknowledgements
We thank [Monodepth2](https://github.com/nianticlabs/monodepth2) and [FalNet](https://github.com/JuanLuisGonzalez/FAL_net) for their outstanding methods and codes.

## 🐒 Citation
If you find our paper or code useful, please cite
```bibtex
@inproceedings{wang2023planedepth,
    author    = {Wang, Ruoyu and Yu, Zehao and Gao, Shenghua},
    title     = {PlaneDepth: Self-Supervised Depth Estimation via Orthogonal Planes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {21425-21434}
}
```

## 🐓 Contact us
If you have any questions, don't hesitate to contact us at wangry3@shanghaitech.edu.cn or open an issue. Let's discuss and create more sparkling✨ works!
