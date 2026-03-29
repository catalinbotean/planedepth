# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import copy
import random

import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter

import json
import torchvision

from utils import *
from kitti_utils import *
from layers import *

import datasets as datasets
import networks
from IPython import embed
        
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed) # torch doc says that torch.manual_seed also work for CUDA
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # When launched via `torchrun` the env vars are already set.
        # For single-GPU / Colab usage (plain `python train.py`) set them here.
        if 'RANK' not in os.environ:
            os.environ.setdefault('RANK',        '0')
            os.environ.setdefault('LOCAL_RANK',  '0')
            os.environ.setdefault('WORLD_SIZE',  '1')
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '12355')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.opt.batch_size = self.opt.batch_size // max(torch.cuda.device_count(), 1)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
        init_seeds(1+self.local_rank)

        if dist.get_rank() == 0:
            save_code("./trainer.py", self.log_path)
            if self.opt.net_type == "ResNet":
                save_code("./networks/depth_decoder.py", self.log_path)
                save_code("./train_ResNet.sh", self.log_path)
            elif self.opt.net_type == "PladeNet":
                save_code("./networks/plade_net.py", self.log_path)
                save_code("./train_PladeNet.sh", self.log_path)
            elif self.opt.net_type == "FalNet":
                save_code("./networks/fal_net.py", self.log_path)
                save_code("./train_FalNet.sh", self.log_path)
            

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        if self.opt.use_mom:
            self.opt.flip_right = True
        
        if self.opt.flip_right:
            self.opt.batch_size = self.opt.batch_size // 2

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda")

        if not self.opt.no_stereo:
            self.target_sides = ["r"] + self.opt.novel_frame_ids
        else:
            self.target_sides = self.opt.novel_frame_ids

        self.models.update(self.create_models())

        # --- Semantic plane gate -----------------------------------------
        # A frozen pretrained segmenter extracts dense class probabilities;
        # a small learnable SemanticPlaneGate maps them to per-family logit
        # biases that steer the plane softmax toward semantically consistent
        # depth families (sky→XY, road→XZ, wall→YZ).
        self.semantic_backbone = None
        if self.opt.use_semantic_gate and self.opt.net_type == "ResNet":
            seg = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=True)
            seg.eval()
            for p in seg.parameters():
                p.requires_grad_(False)
            self.semantic_backbone = seg.to(self.device)

            n_learned = (self.opt.num_learned_families *
                         self.opt.learned_planes_per_family)
            self.models["semantic_gate"] = networks.SemanticPlaneGate(
                num_classes=self.opt.semantic_num_classes,
                n_xy=self.opt.disp_levels,
                n_xz=self.opt.xz_levels,
                n_yz=self.opt.yz_levels,
                n_learned=n_learned,
            )
            # ImageNet normalisation constants (NCHW broadcastable)
            self.sem_mean = torch.tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.sem_std  = torch.tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        if len(self.opt.novel_frame_ids) > 0 and not self.opt.use_colmap:
            self.models["pose_encoder"] = networks.ResnetPoseEncoder(18, True, 2)
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, num_ep=8)
            
        for model_name, model in self.models.items():
            model = model.to(self.device)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.models[model_name] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.parameters_to_train += list(self.models[model_name].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, betas=(self.opt.beta_1, self.opt.beta_2))
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, milestones=self.opt.milestones, gamma=0.5)

        if self.opt.load_weights_folder is not None:
            self.load_model()
            
        if self.opt.self_distillation > 0:
            self.fixed_models = {}
            self.fixed_models["encoder"] = copy.deepcopy(self.models["encoder"].module).eval()
            self.fixed_models["depth"] = copy.deepcopy(self.models["depth"].module).eval()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "./splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // (self.opt.batch_size * torch.cuda.device_count()) * self.opt.num_epochs

        def worker_init(worker_id):
            worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.novel_frame_ids, is_train=True, use_crop=not self.opt.no_crop, use_colmap=self.opt.use_colmap, colmap_path=self.opt.colmap_path, img_ext=img_ext)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, sampler=self.train_sampler, pin_memory=True, drop_last=True, worker_init_fn=worker_init, collate_fn=rmnone_collate)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.novel_frame_ids, is_train=False, use_crop=False, use_colmap=False, img_ext=img_ext)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, sampler=self.val_sampler, pin_memory=True, drop_last=False)

        if self.opt.use_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = BackprojectDepth(self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.height, self.opt.width)
        self.project_3d.to(self.device)
        
        self.homography_warp = HomographyWarp(self.opt.height, self.opt.width)
        self.homography_warp.to(self.device)
        
        if self.opt.pc_net == "vgg19":
            self.pc_net = Vgg19_pc().cuda()
        elif self.opt.pc_net == "resnet18":
            self.pc_net = Resnet18_pc().cuda()
        self.softmax = nn.Softmax(1)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if dist.get_rank() == 0:
            
            self.writers = {}
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))
            self.save_opts()
            
            self.log_file = open(os.path.join(self.log_path, "logs.log"),'w')
            
        self.best_absrel = 10.

    def create_models(self):
        models = {}
        if self.opt.net_type == "ResNet":
            print("train ResNet")
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, True)
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                         self.opt.disp_levels,
                                                         self.opt.disp_min,
                                                         self.opt.disp_max,
                                                         self.opt.num_ep,
                                                         pe_type=self.opt.pe_type,
                                                         use_denseaspp=self.opt.use_denseaspp,
                                                         xz_levels=self.opt.xz_levels,
                                                         yz_levels=self.opt.yz_levels,
                                                         use_mixture_loss=self.opt.use_mixture_loss,
                                                         render_probability=self.opt.render_probability,
                                                         plane_residual=self.opt.plane_residual,
                                                         pixelwise_plane_residual=self.opt.pixelwise_plane_residual,
                                                         use_cross_plane_attn=self.opt.use_cross_plane_attn,
                                                         cross_plane_attn_tau=self.opt.cross_plane_attn_tau,
                                                         adaptive_plane_range=self.opt.adaptive_plane_range,
                                                         adaptive_range_margin=self.opt.adaptive_range_margin,
                                                         num_learned_families=self.opt.num_learned_families,
                                                         learned_planes_per_family=self.opt.learned_planes_per_family,
                                                         use_multiscale_logits=self.opt.use_multiscale_logits)
                
        elif self.opt.net_type == "PladeNet":
            print("train PladeNet")
            self.models["plade"] = networks.PladeNet(False, 
                                                     self.opt.disp_levels,
                                                     self.opt.disp_min,
                                                     self.opt.disp_max, 
                                                     self.opt.num_ep, 
                                                     xz_levels=self.opt.xz_levels, 
                                                     use_mixture_loss=self.opt.use_mixture_loss, 
                                                     render_probability=self.opt.render_probability,
                                                     plane_residual=self.opt.plane_residual)

        elif self.opt.net_type == "FalNet":
            print("train FalNet")
            self.models["fal"] = networks.FalNet(False, self.opt.height, self.opt.width, self.opt.disp_levels, self.opt.disp_min, self.opt.disp_max)
        
        else:
            print("undefined model type")
            quit()
        return models

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        for self.epoch in range(self.opt.start_epoch):
            self.model_lr_scheduler.step()
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self._update_active_plane_levels()
            self.run_epoch()
            if dist.get_rank() == 0:
                self.save_model("last_models")
                
    def _update_active_plane_levels(self):
        """Coarse-to-fine plane annealing: linearly unlock XY plane levels.

        When plane_anneal_start < disp_levels, training begins with only
        plane_anneal_start XY planes active (coarser depth quantisation).
        Over plane_anneal_epochs epochs the active count grows linearly to
        disp_levels.  After that, all planes are unlocked for the remainder
        of training.

        Only applies to ResNet (DepthDecoder).  No-op when
        plane_anneal_start == disp_levels (the default).
        """
        if self.opt.net_type != "ResNet":
            return
        n_start = self.opt.plane_anneal_start
        n_final = self.opt.disp_levels
        if n_start >= n_final:
            return  # annealing disabled
        t = min(self.epoch / max(self.opt.plane_anneal_epochs, 1), 1.0)
        n_active = int(n_start + t * (n_final - n_start))
        depth_model = self.models["depth"]
        m = depth_model.module if hasattr(depth_model, "module") else depth_model
        m.set_active_levels(n_active)
        if dist.get_rank() == 0 and n_active < n_final:
            print("Plane annealing: active XY levels = {}/{}".format(n_active, n_final))

    def add_flip_right_inputs(self, inputs):
        new_inputs = {}
        new_inputs[("color", "l")] = torch.cat([inputs[("color", "l")], inputs[("color", "r")].flip(-1)], dim=0)
        new_inputs[("color", "r")] = torch.cat([inputs[("color", "r")], inputs[("color", "l")].flip(-1)], dim=0)
        new_inputs[("color_aug", "l")] = torch.cat([inputs[("color_aug", "l")], inputs[("color_aug", "r")].flip(-1)], dim=0)
        new_inputs[("color_aug", "r")] = torch.cat([inputs[("color_aug", "r")], inputs[("color_aug", "l")].flip(-1)], dim=0)
        grid_fliped = inputs["grid"].clone()
        grid_fliped[:, 0, :, :] *= -1.
        grid_fliped = grid_fliped.flip(-1)
        new_inputs["grid"] = torch.cat([inputs["grid"], grid_fliped], dim=0)
        new_inputs[("depth_gt", "l")] = torch.cat([inputs[("depth_gt", "l")], inputs[("depth_gt", "r")].flip(-1)], dim=0)
        new_inputs[("depth_gt", "r")] = torch.cat([inputs[("depth_gt", "r")], inputs[("depth_gt", "l")].flip(-1)], dim=0)
        
        new_inputs["K"] = inputs["K"].repeat(2, 1, 1)
        new_inputs["inv_K"] = inputs["inv_K"].repeat(2, 1, 1)
        
        new_inputs[("Rt", "l")] = inputs[("Rt", "l")].repeat(2, 1, 1)
        new_inputs[("Rt", "r")] = inputs[("Rt", "r")].repeat(2, 1, 1)
        
        # The the left +1/-1 frame becomes the right side, but it should not affect the training
        for novel_frame_id in self.opt.novel_frame_ids:
            new_inputs[("color", novel_frame_id)] = torch.cat([inputs[("color", novel_frame_id)], inputs[("color", novel_frame_id)].flip(-1)], dim=0)
            new_inputs[("color_aug", novel_frame_id)] = torch.cat([inputs[("color_aug", novel_frame_id)], inputs[("color_aug", novel_frame_id)].flip(-1)], dim=0)

        return new_inputs

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.train_sampler.set_epoch(self.epoch)
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            if inputs is None:
                self.model_optimizer.zero_grad()
                self.model_optimizer.step()
                self.step += 1
                continue
            
            before_op_time = time.time()
            
            if self.opt.flip_right:
                inputs = self.add_flip_right_inputs(inputs)
            
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss/total_loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % 100 == 0 and self.step < self.opt.log_frequency
            late_phase = self.step % self.opt.log_frequency == 0

            if early_phase or late_phase:
                if dist.get_rank() == 0:
                    self.log_time(batch_idx, duration, losses)

                    losses.update(self.compute_depth_losses(inputs, outputs))

                    self.log("train", losses)

            self.step += 1
            
            if batch_idx == 0 and dist.get_rank() == 0:
                self.log_img("train", inputs, outputs, batch_idx)
                
        self.val()
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        # maybe we need use the same name for different model in self.models
        if self.opt.net_type == "ResNet":
            features = self.models["encoder"](inputs[("color_aug", "l")])
            sem_bias = self._get_semantic_bias(inputs[("color_aug", "l")])
            outputs = self.models["depth"](features, inputs["grid"],
                                           sem_bias=sem_bias)
        elif self.opt.net_type == "PladeNet":
            outputs = self.models["plade"](inputs[("color_aug", "l")], inputs["grid"])
        elif self.opt.net_type == "FalNet":
            outputs = self.models["fal"](inputs[("color_aug", "l")])

        outputs.update(self.predict_poses(inputs))

        self.pred_novel_images(inputs, outputs)
        
        if self.opt.use_mom and inputs[("color", "l")].shape[0] == self.opt.batch_size*2:
            self.mirror_occlusion_mask(outputs)
            
        if self.opt.self_distillation > 0.:
            with torch.no_grad():
                outputs["disp_pp"], outputs["mask_novel"], outputs["teacher_confidence"] = \
                    self.generate_post_process_disp(inputs)
            
        if self.opt.alpha_self > 0.:
            self.pred_self_images(inputs, outputs)
        
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.
        outputs[("Rt", "r")] = inputs[("Rt", "r")]

        for f_i in self.opt.novel_frame_ids:
            
            if not self.opt.use_colmap:
                if f_i < 0:
                    pose_inputs = [inputs[("color_aug", f_i)], inputs[("color_aug", "l")]]
                else:
                    pose_inputs = [inputs[("color_aug", "l")], inputs[("color_aug", f_i)]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                axisangle, translation = self.models["pose"](pose_inputs, inputs["grid"])
                outputs[("axisangle", f_i)] = axisangle
                outputs[("translation", f_i)] = translation

                # Invert the matrix if the frame id is negative
                Rt = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            else:
                Rt = inputs[("Rt", f_i)].float()
                
            Rt_Rc = torch.zeros_like(Rt)
            
            gx0 = (inputs["grid"][:, 0, 0, -1] + inputs["grid"][:, 0, 0, 0]) / 2.
            gy0 = (inputs["grid"][:, 1, -1, 0] + inputs["grid"][:, 1, 0, 0]) / 2.
            f = (inputs["grid"][:, 0, 0, -1] - inputs["grid"][:, 0, 0, 0]) / 2.
            Rc_v = torch.stack([-gx0/(2*0.58), -gy0/(2*1.92), f], dim=1)
            Rc = torch.eye(3).cuda()
            Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
            Rc[:, :, 2] = Rc_v
            outputs[("Rc", f_i)] = Rc
            Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(Rt[:, :3, :3], torch.inverse(Rc)))
            if self.opt.use_colmap:
                Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, Rt[:, :3, 3:4])
                
            outputs[("Rt", f_i)] = Rt_Rc
                
        return outputs
    
    def generate_post_process_disp(self, inputs):
        # self.set_eval()
        input_images = torch.cat([inputs[("color_aug", "l")], inputs[("color_aug", "l")].flip(-1)], dim=0)
        if self.opt.num_ep > 0:
            grid_fliped = inputs["grid"].clone()
            grid_fliped[:, 0, :, :] *= -1.
            grid_fliped = grid_fliped.flip(-1)
            input_grids = torch.cat([inputs["grid"], grid_fliped], dim=0)
        
        if self.opt.net_type == "ResNet":
            features = self.fixed_models["encoder"](input_images)
            outputs = self.fixed_models["depth"](features, input_grids)
        elif self.opt.net_type == "PladeNet":
            outputs = self.models["plade"](input_images, input_grids)
        elif self.opt.net_type == "FalNet":
            outputs = self.models["fal"](input_images)

        B, N, H, W = outputs["probability"].shape
        B = B // 2
        pix_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        pix_coords = torch.stack(pix_coords, dim=0).cuda().float()
        
        pix_coords_r = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
        pix_coords_r[:, :, 0, :, :] += outputs["disp_layered"][:B, ...]
        pix_coords_r[:, :, 0, :, :] /= (W-1)
        pix_coords_r[:, :, 1, :, :] /= (H-1)
        pix_coords_r = (pix_coords_r - 0.5) * 2
        pix_coords_r = pix_coords_r.reshape(B*N, 2, H, W)
        pix_coords_r = pix_coords_r.permute(0, 2, 3, 1)
        
        pix_coords_l = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
        pix_coords_l[:, :, 0, :, :] -= outputs["disp_layered"][B:, ...]
        pix_coords_l[:, :, 0, :, :] /= (W-1)
        pix_coords_l[:, :, 1, :, :] /= (H-1)
        pix_coords_l = (pix_coords_l - 0.5) * 2
        pix_coords_l = pix_coords_l.reshape(B*N, 2, H, W)
        pix_coords_l = pix_coords_l.permute(0, 2, 3, 1)
    
        #pll = outputs_stage1["probability"][:B, ...]
        pl = outputs["logits"][:B, ...].reshape(B*N, 1, H, W)
        plr = F.grid_sample(pl, pix_coords_r, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
        plr = self.softmax(plr)
        plr = plr.reshape(B*N, 1, H, W)
        o_l = F.grid_sample(plr, pix_coords_l, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
        o_l = o_l.sum(1, True)
        o_l[o_l>1] = 1
        
        pfr = outputs["logits"][B:, ...].flip(-1).reshape(B*N, 1, H, W)
        pfrl = F.grid_sample(pfr, pix_coords_l, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
        pfrl = self.softmax(pfrl).reshape(B*N, 1, H, W)
        o_fr = F.grid_sample(pfrl, pix_coords_r, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
        o_fr = o_fr.sum(1, True)
        o_fr[o_fr>1] = 1
    
        mean_disp = outputs["disp"][:B, ...] * 0.5 + outputs["disp"][B:, ...].flip(-1) * 0.5
    
        disp_pp = mean_disp * o_fr + outputs["disp"][:B, ...] * (1 - o_fr)
        disp_pp = disp_pp * o_l + outputs["disp"][-B:, ...].flip(-1) * (1 - o_l)
        
        mask_novel = F.grid_sample(outputs["probability"][:B, ...].reshape(B*N, 1, H, W), pix_coords_r, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
        mask_novel = mask_novel.sum(1, True)
        mask_novel[mask_novel>1] = 1

        # Teacher confidence from mixture model uncertainty (available when use_mixture_loss=True)
        teacher_confidence = None
        if "depth_confidence" in outputs:
            # Average confidence across original and flipped teacher predictions
            conf_orig   = outputs["depth_confidence"][:B, ...]          # B, 1, H, W
            conf_flipped = outputs["depth_confidence"][B:, ...].flip(-1) # B, 1, H, W
            teacher_confidence = (conf_orig + conf_flipped) * 0.5
            teacher_confidence = teacher_confidence.detach()

        return disp_pp.detach(), mask_novel.detach(), teacher_confidence

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        num = 0
        metrics = {}
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                #outputs, losses = self.process_batch(inputs)
                #losses.update(self.compute_depth_losses(inputs, outputs))
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                
                # maybe we need use the same name for different model in self.models
                if self.opt.net_type == "ResNet":
                    features = self.models["encoder"](inputs[("color_aug", "l")])
                    sem_bias = self._get_semantic_bias(inputs[("color_aug", "l")])
                    outputs = self.models["depth"](features, inputs["grid"],
                                                   sem_bias=sem_bias)
                elif self.opt.net_type == "PladeNet":
                    outputs = self.models["plade"](inputs[("color_aug", "l")], inputs["grid"])
                elif self.opt.net_type == "FalNet":
                    outputs = self.models["fal"](inputs[("color_aug", "l")])
                
                losses = self.compute_depth_losses(inputs, outputs)
                B = inputs[("color_aug", "l")].shape[0]
                num += B
                for k,v in losses.items():
                    if k in metrics:
                        metrics[k] += v * B
                    else:
                        metrics[k] = v * B
                
                if batch_idx % self.opt.log_img_frequency == 0 and self.local_rank == 0:
                    self.log_img("val", inputs, outputs, batch_idx)
                del inputs, outputs, losses
            # since the eval batch size is not the same
            # we need to sum them then mean
            num = torch.ones(1).cuda() * num
            dist.all_reduce(num, op=dist.ReduceOp.SUM)
            for k,v in metrics.items():
                dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
                metrics[k] = metrics[k] / num
            if metrics["de/abs_rel"] < self.best_absrel:
                self.best_absrel = metrics["de/abs_rel"]
                if self.local_rank == 0:
                    self.save_model("best_models")
                    
            if self.local_rank == 0:
                self.log("val", metrics)
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in self.depth_metric_names]) + "\\\\")
                #write to log file
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"), file=self.log_file)
                print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in self.depth_metric_names]) + "\\\\", file=self.log_file)
        self.set_train()

    def pred_novel_images(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        
        B, N, H, W = outputs["probability"].shape
        
        source_side = "l"
        
        for target_side in self.target_sides:
            if self.opt.warp_type == "depth_warp":
                disps = outputs["disp_layered"]
                depths = 0.1 * 0.58 * W / disps
                T = inputs[("Rt", target_side)][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4)
                cam_points = self.backproject_depth(depths.reshape(B*N, 1, H, W), inputs["inv_K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4))
                pix_coords = self.project_3d(cam_points, inputs["K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4), T) #BN, H, W, 2
                
            elif self.opt.warp_type == "disp_warp":
                disps = outputs["disp_layered"]
                pix_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
                pix_coords = torch.stack(pix_coords, dim=0).cuda().float()
                pix_coords = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
                if target_side == "l":
                    pix_coords[:, :, 0, :, :] -= disps
                elif target_side == "r":
                    pix_coords[:, :, 0, :, :] += disps
                pix_coords[:, :, 0, :, :] /= (W-1)
                pix_coords[:, :, 1, :, :] /= (H-1)
                pix_coords = (pix_coords - 0.5) * 2
                pix_coords = pix_coords.reshape(B*N, 2, H, W)
                pix_coords = pix_coords.permute(0, 2, 3, 1)
                padding_mask = outputs["padding_mask"][:, :, None, :, :]
            
            elif self.opt.warp_type == "homography_warp":
                T = outputs[("Rt", target_side)][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4)
                K = inputs["K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4)
                inv_K = inputs["inv_K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B*N, 4, 4)
                pix_coords, padding_mask = self.homography_warp(outputs["distance"], outputs["norm"], T, K, inv_K)
                
            if self.opt.match_aug:
                color_name = "color_aug"
            else:
                color_name = "color"
                
            features = torch.cat([inputs[(color_name, source_side)][:, None].expand(-1, N, -1, -1, -1).reshape(B*N, 3, H, W),\
                                    outputs["logits"].reshape(B*N, 1, H, W)], dim=1)
            
            if self.opt.use_mixture_loss:
                features = torch.cat([features, outputs["sigma"].reshape(B*N, 1, H, W)], dim=1)

            rec_features = F.grid_sample(
                features,
                pix_coords,
                padding_mode="zeros",
                align_corners=True).reshape(B, N, -1, H, W)
            
            # only stereo could compute as this.
            rec_features = rec_features * padding_mask
            
            outputs[("rgb_rec_layered", target_side)] = rec_features[:, :, :3, ...]
            outputs[("logit_rec", target_side)] = rec_features[:, :, 3, ...]
            if self.opt.render_probability:
                # We read dists from output since the layered depth of stereo pair is the same.
                # otherwise we should recompute it.
                alpha = 1. - torch.exp(-F.relu(outputs[("logit_rec", target_side)][:, :-1, ...]) * outputs["dists"])
                ones = torch.ones_like(alpha[:, :1, ...])
                alpha = torch.cat([alpha, ones], dim=1)
                probability_rec = alpha * torch.cumprod(torch.cat([ones, 1.-alpha+1e-10], dim=1), dim=1)[:, :-1, ...]
                outputs[("probability_rec", target_side)] = probability_rec
            else:
                outputs[("probability_rec", target_side)] = self.softmax(outputs[("logit_rec", target_side)])
            if self.opt.use_mixture_loss:
                sigma_rec = rec_features[:, :, 4, ...].clone()
                # sigma_rec[sigma_rec==0] = 1.
                sigma_rec = torch.clamp(sigma_rec, 0.01, 1.)
                outputs[("sigma_rec", target_side)] = sigma_rec
                outputs[("pi_rec", target_side)] = pi_rec = outputs[("probability_rec", target_side)]
                weights_rec = pi_rec / sigma_rec
                weights_rec = weights_rec / weights_rec.sum(1, True)
                outputs[("probability_rec", target_side)] = weights_rec
            outputs[("rgb_rec", target_side)] = (outputs[("rgb_rec_layered", target_side)] * outputs[("probability_rec", target_side)][:, :, None]).sum(1)
        
    def pred_self_images(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = outputs["disp"]
        B, N, H, W = outputs["probability"].shape
        
        depth = 0.1 * 0.58 * W / disp
        T = inputs[("Rt", "r")]
        cam_points = self.backproject_depth(depth, inputs["inv_K"])
        pix_coords = self.project_3d(cam_points, inputs["K"], T) #BN, H, W, 2

        if self.opt.match_aug:
            color_name = "color_aug"
        else:
            color_name = "color"
            
        features = inputs[(color_name, "r")]

        rec_features = F.grid_sample(
            features,
            pix_coords,
            padding_mode="border", 
            align_corners=True)
        
        # only stereo could compute as this.
        #rec_features = rec_features * outputs["padding_mask"][:, :, None, ...]

        outputs["self_rec"] = rec_features
        
        
    def mirror_occlusion_mask(self, outputs):
        with torch.no_grad():
            B, N, H, W = outputs["probability"].shape
            B = B // 2
            pll = outputs["probability"][:B, ...]
            prr = outputs["probability"][B:, ...].flip(-1)
            plr = outputs["probability_rec"][:B, ...]
            prl = outputs["probability_rec"][B:, ...].flip(-1)
            
            pl = torch.stack([pll, prl], dim=2).reshape(B*N, 2, H, W)
            pr = torch.stack([prr, plr], dim=2).reshape(B*N, 2, H, W)
            
            pix_coords_r = self.pix_coords_r.expand(B, -1, -1, -1, -1).reshape(B*N, 2, H, W).permute(0, 2, 3, 1)
            o_r = F.grid_sample(
                pl,
                pix_coords_r,
                padding_mode="zeros", align_corners=True).reshape(B, N, 2, H, W)
            o_r = o_r.sum(1)
            o_r = o_r[:, 0] * o_r[:, 1]
            o_r[o_r>1] = 1
            o_r = o_r.unsqueeze(1)
            
            pix_coords_l = self.pix_coords_l.expand(B, -1, -1, -1, -1).reshape(B*N, 2, H, W).permute(0, 2, 3, 1)
            o_l = F.grid_sample(
                pr,
                pix_coords_l,
                padding_mode="zeros", align_corners=True).reshape(B, N, 2, H, W)
            o_l = o_l.sum(1)
            o_l = o_l[:, 0] * o_l[:, 1]
            o_l[o_l>1] = 1
            o_l = o_l.unsqueeze(1)
            
            outputs["mask_novel"] = torch.cat([o_r, o_l.flip(-1)], dim=0)
            outputs["mask_novel"] = outputs["mask_novel"].detach()
        

    def _get_semantic_bias(self, img):
        """Run frozen segmenter + SemanticPlaneGate → plane logit biases.

        img : (B, 3, H, W) in [0, 1] range (may be 2B when flip_right=True)
        Returns (B, N_planes, H_s, W_s) bias tensor, or None when disabled.
        """
        if self.semantic_backbone is None:
            return None
        with torch.no_grad():
            img_norm = (img - self.sem_mean) / self.sem_std
            sem_logits = self.semantic_backbone(img_norm)["out"]  # B, C, H_s, W_s
        gate = self.models["semantic_gate"]
        return gate(sem_logits)   # B, N_planes, H_s, W_s  (upsampled in decoder)

    def compute_surface_normal_loss(self, outputs, inputs):
        """Image-edge-weighted surface normal smoothness loss.

        Surface normals are derived from the predicted depth map via
        central-difference finite differences in 3-D camera space.  Neighbouring
        normals are encouraged to agree within visually homogeneous regions, while
        edges in the colour image suppress the penalty — just like the standard
        disparity smoothness, but operating on orientation rather than depth
        magnitude.

        Advantage over disparity smoothness
        ------------------------------------
        Disparity gradients are scale-dependent: the same metric curvature gives
        a larger gradient at short range than at long range.  Normal differences
        (1 − cos θ) are scale-invariant, so the loss is equally sensitive to
        planar regions at all depths.

        Math
        ----
        From depth D and inv_K, compute camera-space points
            P(u,v) = D(u,v) · K⁻¹ · [u, v, 1]ᵀ

        Central-difference tangents:
            t_u = P(u+1,v) − P(u−1,v)
            t_v = P(u,v+1) − P(u,v−1)

        Normal:  N = normalize( t_u × t_v )   (B, 3, H-2, W-2)

        Edge-aware cosine loss:
            w_u = exp(−γ · |∇_u I|)
            w_v = exp(−γ · |∇_v I|)
            L = mean( w_u · (1 − N(u,v)·N(u+1,v))
                    + w_v · (1 − N(u,v)·N(u,v+1)) )
        """
        depth = outputs["depth"]                # B, 1, H, W
        B, _, H, W = depth.shape
        inv_K = inputs["inv_K"]                 # B, 4, 4

        # --- backproject depth to camera-space 3-D points --------------------
        # reuse the existing BackprojectDepth layer (same H, W)
        cam_pts = self.backproject_depth(depth, inv_K)   # B, 4, H*W
        pts = cam_pts[:, :3, :].reshape(B, 3, H, W)      # B, 3, H, W  (X, Y, Z)

        # --- central-difference tangents (interior pixels only) --------------
        t_u = pts[:, :, 1:-1, 2:] - pts[:, :, 1:-1, :-2]   # B, 3, H-2, W-2
        t_v = pts[:, :, 2:, 1:-1] - pts[:, :, :-2, 1:-1]   # B, 3, H-2, W-2

        # --- cross product N = t_u × t_v -------------------------------------
        nx = t_u[:, 1] * t_v[:, 2] - t_u[:, 2] * t_v[:, 1]
        ny = t_u[:, 2] * t_v[:, 0] - t_u[:, 0] * t_v[:, 2]
        nz = t_u[:, 0] * t_v[:, 1] - t_u[:, 1] * t_v[:, 0]
        norm = (nx.pow(2) + ny.pow(2) + nz.pow(2) + 1e-8).sqrt()
        normals = torch.stack([nx, ny, nz], dim=1) / norm.unsqueeze(1)  # B, 3, H-2, W-2

        # --- image-edge-aware weights ----------------------------------------
        img = inputs[("color", "l")][:, :, 1:-1, 1:-1]         # B, 3, H-2, W-2
        grad_u = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(1, keepdim=True)  # B,1,H-2,W-3
        grad_v = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(1, keepdim=True)  # B,1,H-3,W-2

        w_u = torch.exp(-self.opt.gamma_smooth * grad_u)   # B, 1, H-2, W-3
        w_v = torch.exp(-self.opt.gamma_smooth * grad_v)   # B, 1, H-3, W-2

        # --- cosine dissimilarity between neighbouring normals ---------------
        # dot product along neighbouring pairs, interior of normals grid
        dot_u = (normals[:, :, :, :-1] * normals[:, :, :, 1:]).sum(1, keepdim=True)   # B,1,H-2,W-3
        dot_v = (normals[:, :, :-1, :] * normals[:, :, 1:, :]).sum(1, keepdim=True)   # B,1,H-3,W-2

        loss_u = (1.0 - dot_u) * w_u
        loss_v = (1.0 - dot_v) * w_v

        return loss_u.mean() + loss_v.mean()

    def compute_lr_consistency_loss(self, outputs):
        """Left-right disparity geometric consistency loss.

        When flip_right=True the batch contains:
          outputs["disp"][:B]  – left-camera disparity  (B, 1, H, W)
          outputs["disp"][B:]  – right-camera disparity in the flipped coordinate
                                  system, i.e. disp[B:].flip(-1) is in left-camera coords.

        Consistency requirement at each left pixel u:
            d_L(u) ≈ d_R( u + d_L(u) )

        i.e. warp the right disparity map toward the left using d_L as the
        shift, and compare with d_L.  Pixels that fall out of the image
        boundary (occluded/truncated) are masked out automatically by
        padding_mode="zeros".

        Optional confidence weighting: if the mixture model variance is
        available (outputs["disp_var"]), use it to downweight uncertain
        regions of the *left* prediction (the query side).
        """
        B_full = outputs["disp"].shape[0]
        B = B_full // 2
        H, W = outputs["disp"].shape[2], outputs["disp"].shape[3]

        d_L = outputs["disp"][:B]          # B, 1, H, W  (pixel-unit disparities)
        d_R = outputs["disp"][B:].flip(-1) # B, 1, H, W  (back to left-camera coords)

        # Build sampling grid: for each left pixel (u, v), sample d_R at (u + d_L, v).
        # grid_sample expects normalised coordinates in [-1, 1].
        ys = torch.linspace(-1.0, 1.0, H, device=d_L.device)
        xs = torch.linspace(-1.0, 1.0, W, device=d_L.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")       # H, W
        grid_x = grid_x[None, None].expand(B, 1, -1, -1)             # B, 1, H, W
        grid_y = grid_y[None, None].expand(B, 1, -1, -1)             # B, 1, H, W

        # Convert pixel-unit d_L to normalised shift: +2*d_L/(W-1)
        d_L_norm = d_L * (2.0 / max(W - 1, 1))
        sample_x = (grid_x + d_L_norm).clamp(-1.0, 1.0)             # B, 1, H, W
        sample_grid = torch.stack([sample_x[:, 0], grid_y[:, 0]], dim=-1)  # B, H, W, 2

        # Sample right disparity at shifted positions.
        # padding_mode="zeros" → out-of-bounds pixels return 0, which we use as mask.
        d_R_warped = F.grid_sample(
            d_R, sample_grid, padding_mode="zeros", align_corners=True)  # B, 1, H, W

        # Validity mask: only pixels where d_R had a valid sample
        with torch.no_grad():
            valid = (d_R_warped > 0).float()

        lr_error = torch.abs(d_L - d_R_warped)

        # Optional: confidence-weight by inverse mixture variance of the left prediction
        if "disp_var" in outputs:
            conf = 1.0 / (outputs["disp_var"][:B] + 1e-4)
            conf = conf / (conf.mean() + 1e-8)
            lr_error = lr_error * conf

        lr_loss = (lr_error * valid).sum() / (valid.sum() + 1e-8)
        return lr_loss

    def perceptual_loss(self, pred, target, source=None):
        pred_vgg = self.pc_net(pred)
        target_vgg = self.pc_net(target)
        if source is not None:
            source_vgg = self.pc_net(source)
        
        loss_pc = 0
        for i in range(3):
            l_p = ((pred_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
            if source is not None:#automask
                l_p_auto = ((source_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
                l_p, _ = torch.cat([l_p, l_p_auto], dim=1).min(1, True)
            loss_pc += l_p.mean()
        return loss_pc

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.use_ssim:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        else:
            reprojection_loss = l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        B, N, H, W = outputs["probability"].shape
        losses = {}
        losses["loss/ph_loss"] = 0
        losses["loss/pc_loss"] = 0
        if self.opt.alpha_self > 0.:
            losses["loss/self_loss"] = 0
        losses["loss/total_loss"] = 0
        if self.opt.alpha_lr_consistency > 0.:
            losses["loss/lr_consistency_loss"] = 0
        if self.opt.alpha_entropy > 0.:
            losses["loss/entropy"] = 0
        if self.opt.alpha_normal_smooth > 0.:
            losses["loss/normal_smooth_loss"] = 0

        if self.opt.match_aug:
            color_name = "color_aug"
        else:
            color_name = "color"

        for target_side in self.target_sides:
            total_loss = 0

            pred = outputs[("rgb_rec", target_side)]
            
            target = inputs[(color_name, target_side)]

            if "mask_novel" in outputs.keys():
                mask = outputs["mask_novel"]
                pred = pred*mask+target*(1.-mask)
            
            if self.opt.use_mixture_loss:
                error = torch.abs(outputs[("rgb_rec_layered", target_side)] - target[:, None]).mean(2)
                ph_loss = multimodal_loss(error, outputs[("sigma_rec", target_side)], outputs[("pi_rec", target_side)], dist='lap')#.mean()
                if self.opt.automask:
                    error_auto = torch.abs(inputs[(color_name, "l")][:, None] - target[:, None]).mean(2)
                    ph_loss_auto = multimodal_loss(error_auto, outputs[("sigma_rec", target_side)].detach(), outputs[("pi_rec", target_side)].detach(), dist='lap')
                    ph_loss, _ = torch.cat([ph_loss, ph_loss_auto], dim=1).min(1, True)
                if "mask_novel" in outputs.keys():
                    ph_loss = ph_loss * mask
            else:
                ph_loss = torch.abs(pred - target).mean(1, True)
                if self.opt.automask:
                    ph_loss_auto = torch.abs(inputs[(color_name, "l")] - target).mean(1, True)
                    ph_loss, _ = torch.cat([ph_loss, ph_loss_auto], dim=1).min(1, True)
            # Focal photometric loss: upweight hard pixels (large reconstruction
            # error), downweight easy pixels.  gamma=0 → standard L1/mixture.
            # w_focal = (err / mean_err)^gamma, clamped to avoid extreme weights.
            if self.opt.focal_ph_gamma > 0.:
                with torch.no_grad():
                    w_focal = (ph_loss.detach() /
                               (ph_loss.detach().mean() + 1e-8)
                               ).pow(self.opt.focal_ph_gamma).clamp(max=4.0)
                ph_loss = (w_focal * ph_loss).mean()
            else:
                ph_loss = ph_loss.mean()
            losses["loss/ph_loss"] += ph_loss
            total_loss += ph_loss

            if not self.opt.automask:
                pc_loss = self.perceptual_loss(pred, target).mean()
            else:
                pc_loss = self.perceptual_loss(pred, target, inputs[(color_name, "l")]).mean()
            losses["loss/pc_loss"] += pc_loss
            total_loss += self.opt.alpha_pc * pc_loss
            
            if self.opt.alpha_self > 0.:
                self_loss = self.compute_reprojection_loss(outputs[("self_rec", target_side)], inputs[(color_name, "l")]).mean()
                losses["loss/self_loss"] += self_loss
                total_loss += self.opt.alpha_self * self_loss
                
            if self.opt.self_distillation > 0:
                disp_loss = torch.abs(outputs["disp"] - outputs["disp_pp"])
                if self.opt.uncertainty_weighted_distillation and outputs.get("teacher_confidence") is not None:
                    conf = outputs["teacher_confidence"]           # B, 1, H, W
                    # Normalise so the effective batch size is preserved
                    conf = conf / (conf.mean() + 1e-8)
                    disp_loss = (disp_loss * conf).mean()
                else:
                    disp_loss = disp_loss.mean()
                losses["loss/disp_loss"] = disp_loss
                total_loss += self.opt.self_distillation * disp_loss
            
            losses["loss/total_loss"] += total_loss
            
        for k, v in losses.items():
            v /= len(self.target_sides)
            
        # Confidence-weighted smoothness: when the mixture model provides
        # disp_var, amplify smoothness in high-confidence regions and relax it
        # near uncertain pixels (boundaries, occlusions, sky).
        disp_crop = outputs["disp"][..., int(0.2 * W):]
        img_crop  = inputs[("color", "l")][..., int(0.2 * W):]
        if "disp_var" in outputs and self.opt.use_confidence_smooth:
            conf_crop = outputs["disp_var"][..., int(0.2 * W):]
            # disp_var = variance → invert for confidence
            conf_crop = 1.0 / (conf_crop + 1e-4)
            smooth_loss = get_smooth_loss_disp_confidence(
                disp_crop, img_crop, conf_crop, gamma=self.opt.gamma_smooth)
        else:
            smooth_loss = get_smooth_loss_disp(
                disp_crop, img_crop, gamma=self.opt.gamma_smooth)
        losses["loss/smooth_loss"] = smooth_loss

        losses["loss/total_loss"] += self.opt.alpha_smooth * smooth_loss

        # Left-right geometric consistency loss (requires flip_right so both
        # hemispheres are present in outputs["disp"])
        if self.opt.alpha_lr_consistency > 0. and self.opt.flip_right:
            lr_loss = self.compute_lr_consistency_loss(outputs)
            losses["loss/lr_consistency_loss"] = lr_loss
            losses["loss/total_loss"] += self.opt.alpha_lr_consistency * lr_loss

        # Surface normal smoothness: image-edge-weighted cosine loss on normals
        # derived from the depth map via finite differences in 3-D space.
        # Scale-invariant complement to the disparity smoothness loss.
        if self.opt.alpha_normal_smooth > 0.:
            normal_loss = self.compute_surface_normal_loss(outputs, inputs)
            losses["loss/normal_smooth_loss"] = normal_loss
            losses["loss/total_loss"] += self.opt.alpha_normal_smooth * normal_loss

        # Mixture entropy regularization: penalise high-entropy (uniform) plane
        # distributions.  H(π) = -Σ_k π_k log π_k ≥ 0; minimising H forces the
        # network to commit to a single plane rather than hedging.
        # We use "pi" (the soft, pre-sigma-rescaled weights) so the gradient
        # flows through the raw softmax, not through the sigma normalisation.
        if self.opt.alpha_entropy > 0.:
            pi = outputs.get("pi", outputs["probability"])   # B, N, H, W
            # clamp for numerical safety before log
            entropy = -(pi * (pi + 1e-8).log()).sum(1).mean()
            losses["loss/entropy"] = entropy
            losses["loss/total_loss"] += self.opt.alpha_entropy * entropy

        return losses

    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs["depth"].detach()
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80).detach()
        depth_pred = depth_pred * 2. / (inputs["grid"][:, 0:1, :, -1:] - inputs["grid"][:, 0:1, :, 0:1])
        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        
        depth_gt = inputs[("depth_gt", "l")]
        B, _, H, W = depth_gt.shape

        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, int(0.40810811 * H):int(0.99189189 * H), int(0.03594771 * W):int(0.96405229 * W)] = 1
        mask = mask * crop_mask

        depth_gt = torch.clamp(depth_gt[mask], 1e-3, 80)
        depth_pred = depth_pred[mask]
        if self.opt.no_stereo:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        else:
            depth_pred *= 5.4

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            #losses[metric] = np.array(depth_errors[i].cpu())
            losses[metric] = depth_errors[i]
        return losses

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size * torch.cuda.device_count() / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss/total_loss"].cpu().data,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(l, v, self.step)
        # Log how open the semantic gate is (tanh(gate_scale) ∈ (-1, 1))
        if self.semantic_backbone is not None and "semantic_gate" in self.models:
            gate = self.models["semantic_gate"]
            m = gate.module if hasattr(gate, "module") else gate
            gate_strength = torch.tanh(m.gate_scale).item()
            writer.add_scalar("semantic/gate_strength", gate_strength, self.step)
                    
    def log_img(self, mode, inputs, outputs, val_idx):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in ["l", "r"] + self.opt.novel_frame_ids:
                writer.add_image(
                    "color_{}/{}".format(frame_id, self.epoch),
                    inputs[("color", frame_id)][j].data, val_idx+j)
                
            if mode == "train":
                
                for frame_id in self.target_sides:
                    writer.add_image(
                        "color_pred_{}/{}".format(frame_id, self.epoch),
                        outputs[("rgb_rec", frame_id)][j].data, val_idx+j)
                    
                if "disp_pp" in outputs:
                    writer.add_image(
                        "disp_pp/{}".format(self.epoch),
                        normalize_image(outputs["disp_pp"][j]), val_idx+j)
                if "disp_var" in outputs:
                    writer.add_image(
                        "disp_var/{}".format(self.epoch),
                        normalize_image(outputs["disp_var"][j]), val_idx+j)

            writer.add_image(
                "disp/{}".format(self.epoch),
                normalize_image(outputs["disp"][j]), val_idx+j)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.log_path
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.module.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].module.state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].module.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
