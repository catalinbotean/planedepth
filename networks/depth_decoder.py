# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .denseaspp import DenseAspp

from collections import OrderedDict
from layers import *


class CrossPlaneAttention(nn.Module):
    """Cross-attention between XY, XZ, and YZ plane families.

    PlaneDepth's three families are geometrically coupled: at pixel (u, v),
    XY plane k at depth Z_k uniquely determines the camera-space height Y and
    lateral position X.  That height must equal the height of the dominant XZ
    plane, and that X must match the dominant YZ plane.  The current decoder
    ignores this coupling — each family votes independently.

    This module enforces consistency by using the disparity of one family's
    planes to *bias* the logits of another family, so geometrically compatible
    plane pairs are jointly up-weighted.

    Compatibility is computed in disparity space (simpler and more general than
    physical space): two planes are compatible at a pixel if they predict the
    same disparity there.

        C[k, j, s] = exp( -( d_A[k, s] - d_B[j, s] )^2 / tau^2 )

    where s is the relevant spatial coordinate (row for XY↔XZ, col for XY↔YZ).

    Cross-attention refinement for family B (query) from family A (key):

        attn[j, k, s] = softmax_k( logit_A[k, s] + log C[k, j, s] )
        delta[j, s]   = sum_k  attn[j, k, s] * logit_A[k, s]
        logit_B'[j]   = logit_B[j] + tanh(alpha) * delta[j]

    All alpha scalars are zero-initialised so the module starts as identity
    and the base model can fully recover during early training.
    """

    def __init__(self, n_xy, n_xz, n_yz, tau=1.0):
        super(CrossPlaneAttention, self).__init__()
        self.n_xy = n_xy
        self.n_xz = n_xz
        self.n_yz = n_yz
        self.tau = tau

        # One learnable scalar per attention direction, zero-init → identity at start
        if n_xz > 0:
            self.alpha_xy_to_xz = nn.Parameter(torch.zeros(1))
            self.alpha_xz_to_xy = nn.Parameter(torch.zeros(1))
        if n_yz > 0:
            self.alpha_xy_to_yz = nn.Parameter(torch.zeros(1))
            self.alpha_yz_to_xy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def _compat(self, disp_a, disp_b):
        """Disparity-space compatibility between two plane families.

        Args:
            disp_a: (B, N_a, S) — disparities for family A along spatial dim S
            disp_b: (B, N_b, S) — disparities for family B along spatial dim S

        Returns:
            C: (B, N_a, N_b, S)  where C[b, a, b_idx, s] = exp(-Δd² / τ²)
        """
        # (B, N_a, 1, S) - (B, 1, N_b, S)  →  (B, N_a, N_b, S)
        diff = disp_a[:, :, None, :] - disp_b[:, None, :, :]
        return torch.exp(-diff.pow(2) / (self.tau ** 2))

    # ------------------------------------------------------------------
    # Core attention step
    # ------------------------------------------------------------------

    def _cross_attend(self, logits_q, logits_k, compat, alpha, spatial_dim):
        """Refine query-family logits using key-family logits + geometric bias.

        Args:
            logits_q   : (B, N_q, H, W) — family being refined
            logits_k   : (B, N_k, H, W) — family providing evidence
            compat     : (B, N_k, N_q, S) — C[b, k, j, s]; S = H or W
            alpha      : scalar nn.Parameter
            spatial_dim: 'h' (XZ — varies with row) or 'w' (YZ — varies with col)

        Returns refined logits_q of the same shape.
        """
        if spatial_dim == 'h':
            # Collapse columns: key logits averaged over W  →  (B, N_k, H)
            lk = logits_k.mean(-1)                             # B, N_k, H
            # compat: (B, N_k, N_q, H) → permute → (B, N_q, N_k, H)
            C = compat.permute(0, 2, 1, 3)                     # B, N_q, N_k, H
            # Attention logit for query-plane j, key-plane k, row v:
            #   lk[b,k,v]  +  log C[b,j,k,v]
            attn = lk[:, None, :, :] + torch.log(C + 1e-8)    # B, N_q, N_k, H
            attn = torch.softmax(attn, dim=2)                  # B, N_q, N_k, H
            delta = (attn * lk[:, None, :, :]).sum(2)          # B, N_q, H
            delta = delta.unsqueeze(-1).expand_as(logits_q)    # B, N_q, H, W
        else:  # 'w'
            # Collapse rows: key logits averaged over H  →  (B, N_k, W)
            lk = logits_k.mean(-2)                             # B, N_k, W
            C = compat.permute(0, 2, 1, 3)                     # B, N_q, N_k, W
            attn = lk[:, None, :, :] + torch.log(C + 1e-8)    # B, N_q, N_k, W
            attn = torch.softmax(attn, dim=2)                  # B, N_q, N_k, W
            delta = (attn * lk[:, None, :, :]).sum(2)          # B, N_q, W
            delta = delta.unsqueeze(-2).expand_as(logits_q)    # B, N_q, H, W

        return logits_q + torch.tanh(alpha) * delta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits, disp_layered, padding_mask):
        """
        Args:
            logits       : (B, N_total, H, W) — raw logits from dispconv,
                           already multiplied by padding_mask
            disp_layered : (B, N_total, H, W) — disparity per plane per pixel
            padding_mask : (B, N_total, H, W) — validity mask (0 = invalid pixel
                           for that plane family, e.g. above-horizon for XZ)

        Returns refined logits of the same shape, re-masked.
        """
        n_xy, n_xz, n_yz = self.n_xy, self.n_xz, self.n_yz

        l_xy = logits[:, :n_xy]                                # B, N_xy, H, W
        l_xz = logits[:, n_xy:n_xy + n_xz] if n_xz > 0 else None
        l_yz = logits[:, n_xy + n_xz:]     if n_yz > 0 else None

        # XY planes are fronto-parallel: their disparity is constant over (H, W).
        # Take the value at position (0, 0) as representative.
        disp_xy_scalar = disp_layered[:, :n_xy, 0, 0]         # B, N_xy

        # --- XY ↔ XZ (ground planes vary only with image row) --------
        if n_xz > 0:
            # XZ disparity at each row (averaged over W for stability)
            disp_xz_row = disp_layered[:, n_xy:n_xy + n_xz, :, 0]  # B, N_xz, H

            # XY disparity broadcast to each row
            disp_xy_row = disp_xy_scalar[:, :, None].expand(
                -1, -1, disp_xz_row.shape[-1])                # B, N_xy, H

            C_xy_xz = self._compat(disp_xy_row, disp_xz_row) # B, N_xy, N_xz, H

            # Save original logits so both refinements use the un-updated values
            l_xy_orig = l_xy
            l_xz_orig = l_xz

            # Refine XZ with XY evidence (XZ is the query, XY is the key)
            l_xz = self._cross_attend(
                l_xz_orig, l_xy_orig, C_xy_xz,
                self.alpha_xy_to_xz, spatial_dim='h')

            # Refine XY with XZ evidence (XY is the query, XZ is the key)
            # Compatibility is the transpose: C_xz_xy[b, k, j, h] = C_xy_xz[b, j, k, h]
            C_xz_xy = C_xy_xz.permute(0, 2, 1, 3)            # B, N_xz, N_xy, H
            l_xy = self._cross_attend(
                l_xy_orig, l_xz_orig, C_xz_xy,
                self.alpha_xz_to_xy, spatial_dim='h')

        # --- XY ↔ YZ (lateral planes vary only with image column) ----
        if n_yz > 0:
            # YZ disparity at each column (averaged over H for stability)
            disp_yz_col = disp_layered[:, n_xy + n_xz:, :, :].mean(-2)  # B, N_yz, W

            disp_xy_col = disp_xy_scalar[:, :, None].expand(
                -1, -1, disp_yz_col.shape[-1])                # B, N_xy, W

            C_xy_yz = self._compat(disp_xy_col, disp_yz_col) # B, N_xy, N_yz, W

            l_xy_for_yz = l_xy  # use XY as updated by XZ step above
            l_yz_orig   = l_yz

            l_yz = self._cross_attend(
                l_yz_orig, l_xy_for_yz, C_xy_yz,
                self.alpha_xy_to_yz, spatial_dim='w')

            C_yz_xy = C_xy_yz.permute(0, 2, 1, 3)            # B, N_yz, N_xy, W
            l_xy = self._cross_attend(
                l_xy_for_yz, l_yz_orig, C_yz_xy,
                self.alpha_yz_to_xy, spatial_dim='w')

        # Reassemble and re-apply padding mask
        parts = [l_xy]
        if n_xz > 0:
            parts.append(l_xz)
        if n_yz > 0:
            parts.append(l_yz)

        refined = torch.cat(parts, dim=1)
        return refined * padding_mask


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc,
                 no_levels=49,
                 disp_min=2,
                 disp_max=300,
                 num_ep=0,
                 pe_type="neural",
                 use_skips=True,
                 use_denseaspp=True,
                 xz_levels=0,
                 xz_min=0.1852, xz_max=0.3704, #xz_min=0.2315, xz_max=0.3426,#debugxz_min=0.001, xz_max=0.3704,#debug
                 yz_levels=0,
                 yz_min=0.1, yz_max=10.,
                 use_mixture_loss=False,
                 render_probability=False,
                 plane_residual=False,
                 use_cross_plane_attn=False,
                 cross_plane_attn_tau=1.0):
        super(DepthDecoder, self).__init__()

        self.no_levels = no_levels
        self.xz_levels = xz_levels
        self.yz_levels = yz_levels
        self.all_levels = self.no_levels + self.xz_levels + self.yz_levels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.xz_min = xz_min
        self.xz_max = xz_max
        self.yz_min = yz_min
        self.yz_max = yz_max
        self.num_ep = num_ep
        self.pe_type = pe_type
        self.use_mixture_loss = use_mixture_loss
        self.render_probability = render_probability
        self.plane_residual = plane_residual

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.use_denseaspp = use_denseaspp

        print("use {} xy planes, {} xz planes and {} yz planes.".format(self.no_levels, self.xz_levels, self.yz_levels))
        
        # decoder
        self.convs = OrderedDict()
        
        if self.num_ep > 0:
            if self.pe_type == "neural":
                self.convs["epconv"] = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True),
                nn.Conv2d(16, self.num_ep, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True)
                )
            elif self.pe_type == "frequency":
                self.convs["epconv"] = get_embedder((self.num_ep//2 - 1)//2)
        
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1]+self.num_ep if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if i > 0:
                num_ch_in += self.num_ep
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        if use_denseaspp:
            print("use DenseAspp Block")
            self.convs["denseaspp"] = DenseAspp()
            
        if render_probability:
            self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels - 1)
        else:
            self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels)
        
        if self.use_mixture_loss:
            print("use mixture Lap loss")
            self.convs["sigmaconv"] = Conv3x3(self.num_ch_dec[0], self.all_levels)
            
        
            
        if self.plane_residual:
            print("use plane residual")
            self.convs["residualconv"] = nn.Sequential(nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], 1),
                                                       nn.AdaptiveAvgPool2d((1, 1)),
                                                       nn.Conv2d(self.num_ch_dec[0], self.all_levels, 1))


        # self.convs["angleconv"] = nn.Conv2d(self.num_ch_dec[0], 1, 3)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

        # Cross-plane attention: enforces geometric consistency between plane families.
        # Only active when at least one auxiliary family (XZ or YZ) is enabled.
        self.use_cross_plane_attn = use_cross_plane_attn and (xz_levels > 0 or yz_levels > 0)
        if self.use_cross_plane_attn:
            print("use CrossPlaneAttention (tau={})".format(cross_plane_attn_tau))
            self.cross_plane_attn = CrossPlaneAttention(
                n_xy=self.no_levels,
                n_xz=self.xz_levels,
                n_yz=self.yz_levels,
                tau=cross_plane_attn_tau,
            )
        

    def forward(self, input_features, input_grids=None):
        self.outputs = {}
        
        if self.num_ep > 0:
            grids_ep = self.convs["epconv"](input_grids)

        # decoder
        x = input_features[-1]
        if self.num_ep > 0:
            dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
            x = torch.cat([x, dgrid], dim=1)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if self.num_ep > 0 and i > 0:
                dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
                x = torch.cat([x, dgrid], dim=1)
            x = self.convs[("upconv", i, 1)](x)
            
            if i == 4 and self.use_denseaspp:
                x = self.convs["denseaspp"](x)
                
        # angle = (self.sigmoid(self.convs["angleconv"](x).mean(dim=-1).mean(dim=-1)) - 0.5) * 0.75 * np.pi

        B, _, H, W = x.shape
        disp_levels = torch.arange(self.no_levels).cuda()[None, :, None, None]
        disp_levels = disp_levels.expand(B, -1, -1, -1)
        if self.plane_residual:
            residual_levels = self.sigmoid(self.convs["residualconv"](x)) - 0.5 #B, N, 1, 1
            disp_levels = disp_levels + residual_levels[:, :self.no_levels, ...]
        disp_layered = self.disp_max * (self.disp_min / self.disp_max)**(disp_levels / (self.no_levels-1)) # B, N, 1, 1
        distance = 0.1 * 0.58 * W / disp_layered[:, :, 0, 0]
        norm = torch.tensor([0, 0, 1]).cuda()[None, None, :].expand(B, self.no_levels, -1)
        disp_layered = disp_layered.expand(-1, -1, H, W)
        padding_mask = torch.ones_like(disp_layered)
        if self.xz_levels > 0:
            ground_levels = torch.arange(self.xz_levels).cuda()[None, :, None, None]
            ground_levels = ground_levels.expand(B, -1, -1, -1)
            if self.plane_residual:
                ground_levels = ground_levels + residual_levels[:, self.no_levels:self.no_levels+self.xz_levels, ...]
            ground_layered = self.xz_min + (self.xz_max - self.xz_min) * ground_levels / (self.xz_levels-1)
            h = ground_layered[:, :, 0, 0]
            ground_layered = ground_layered.expand(-1, -1, H, W)
            y_grids = input_grids[:, 1:, ...].clone()
            
            xz_padding_mask = (y_grids>=1e-7).expand(-1, self.xz_levels, -1, -1)
            
            y_grids[y_grids<1e-7] = 1e-7
            ground_layered = ground_layered * 1.92 / (y_grids / 2.)
            ground_layered = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * ground_layered
            
            # angle
            # y = torch.linspace(-1, 1, 384).cuda()
            # y_idv =  (y[None, None, :] + torch.tan(-angle)[..., None])
            # y_idv[y_idv<1e-7] = 1e-7
            # ground_layered = h[:, :, None].expand(-1, -1, 384) /y_idv
            # ground_layered = ground_layered[..., None].expand(-1, -1, -1, 1280)
            
            ground_layered = 0.1 * 0.58 * W / ground_layered
            disp_layered = torch.cat([disp_layered, ground_layered], dim=1)
            padding_mask = torch.cat([padding_mask, xz_padding_mask], dim=1)
            
            # original:
            # dgx = input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]
            # dgy = input_grids[:, 1, -1, 0] - input_grids[:, 1, 0, 0]
            # gy_min = input_grids[:, 1, 0, 0]
            # ground_angle = torch.arctan(-(gy_min + 0.5 * dgy) / (1.92 * dgy))
            # norm_angle = ground_angle + np.pi / 2
            # xz_norm = torch.stack([torch.zeros_like(norm_angle), torch.sin(norm_angle), torch.cos(norm_angle)], dim=1)
            # xz_norm = xz_norm[:, None, :].expand(-1, self.xz_levels, -1)
            # norm = torch.cat([norm, xz_norm], dim=1)
            # xz_distance = h * torch.sin(norm_angle)[:, None] * dgx[:, None] / dgy[:, None]
            # distance = torch.cat([distance, xz_distance], dim=1)
            # paper:
            gyc = (input_grids[:, 1, -1, 0] + input_grids[:, 1, 0, 0]) / 2
            py = (gyc + 1) * H / 2
            fs = (input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]) / 2.
            py_cy_fys = (py - H/2) / (H * 1.92 * fs)
            xz_norm = torch.stack([torch.zeros_like(py_cy_fys), torch.ones_like(py_cy_fys), py_cy_fys*torch.ones_like(py_cy_fys)], dim=1)
            xz_normalize = 1 / ((1+py_cy_fys**2)**0.5)
            xz_norm = xz_norm * xz_normalize[:, None]
            xz_distance = h * xz_normalize[:, None]
            xz_norm = xz_norm[:, None, :].expand(-1, self.xz_levels, -1)
            norm = torch.cat([norm, xz_norm], dim=1)
            distance = torch.cat([distance, xz_distance], dim=1)
            
        if self.yz_levels > 0:
            yz_levels = torch.arange(self.yz_levels//2).cuda()[None, :, None, None] #1, 0.5N, 1, 1
            yz_levels = torch.cat([yz_levels, yz_levels], dim=1) #1, N, 1, 1
            yz_levels = yz_levels.expand(B, -1, -1, -1) #B, N, 1, 1
            if self.plane_residual:
                yz_levels = yz_levels + residual_levels[:, -self.yz_levels:, ...]
            yz_disp_max = 1. / self.yz_min
            yz_disp_min = 1. / self.yz_max
            yz_disp_layered = yz_disp_max * (yz_disp_min / yz_disp_max)**(yz_levels / (0.5*self.yz_levels-1)) # B, N, 1, 1
            yz_layered = 1. / yz_disp_layered
            h = yz_layered[:, :, 0, 0]
            
            yz_layered_r = yz_layered[:, :self.yz_levels//2, ...].expand(-1, -1, H, W)
            x_grids_r = input_grids[:, :1, ...].clone()
            yz_padding_mask_r = (x_grids_r>=1e-7).expand(-1, self.yz_levels // 2, -1, -1)
            x_grids_r[x_grids_r<1e-7] = 1e-7
            yz_layered_r = yz_layered_r * 0.58 / (x_grids_r / 2.)
            yz_layered_r = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_layered_r
            
            yz_layered_l = yz_layered[:, -self.yz_levels//2:, ...].expand(-1, -1, H, W)
            x_grids_l = input_grids[:, :1, ...].clone()
            yz_padding_mask_l = (x_grids_l<=-1e-7).expand(-1, self.yz_levels // 2, -1, -1)
            x_grids_l[x_grids_l>-1e-7] = -1e-7
            yz_layered_l = -yz_layered_l * 0.58 / (x_grids_l / 2.)
            yz_layered_l = (input_grids[:, :1, :, -1:] - input_grids[:, :1, :, :1]) / 2. * yz_layered_l
            
            yz_layered = torch.cat([yz_layered_r, yz_layered_l], dim=1)
            yz_layered = 0.1 * 0.58 * W / yz_layered
            disp_layered = torch.cat([disp_layered, yz_layered], dim=1)
            padding_mask = torch.cat([padding_mask, yz_padding_mask_r, yz_padding_mask_l], dim=1)
            
            # All normals face outward
            gxc = (input_grids[:, 0, 0, -1] + input_grids[:, 0, 0, 0]) / 2
            px = (gxc + 1) * W / 2
            fs = (input_grids[:, 0, 0, -1] - input_grids[:, 0, 0, 0]) / 2.
            px_cx_fxs = (px - W/2) / (W * 0.58 * fs)
            yz_norm = torch.stack([torch.ones_like(px_cx_fxs), torch.zeros_like(px_cx_fxs), px_cx_fxs*torch.ones_like(px_cx_fxs)], dim=1)
            yz_normalize = 1 / ((1+px_cx_fxs**2)**0.5)
            yz_norm = yz_norm * yz_normalize[:, None]
            yz_distance = h * yz_normalize[:, None]
            yz_norm_r = yz_norm[:, None, :].expand(-1, self.yz_levels // 2, -1)
            yz_norm_l = -yz_norm[:, None, :].expand(-1, self.yz_levels // 2, -1)
            norm = torch.cat([norm, yz_norm_r, yz_norm_l], dim=1)
            distance = torch.cat([distance, yz_distance], dim=1)

        self.outputs["distance"] = distance
        self.outputs["norm"] = norm
        self.outputs["disp_layered"] = disp_layered
        self.outputs["padding_mask"] = padding_mask
        logits = self.convs["dispconv"](x)
        logits = logits * padding_mask
        if self.use_cross_plane_attn:
            logits = self.cross_plane_attn(logits, disp_layered, padding_mask)
        self.outputs["logits"] = logits
        if self.render_probability:
            depth_layered = 0.1 * 0.58 * W / disp_layered
            dists = depth_layered[:, 1:, ...] - depth_layered[:, :-1, ...]
            # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=1)
            camera_plane = create_camera_plane(height=H, width=W)
            dists = dists * torch.linalg.norm(camera_plane, dim=1, keepdim=True)
            self.outputs["dists"] = dists
            alpha = 1. - torch.exp(-F.relu(self.outputs["logits"]) * dists)
            ones = torch.ones_like(alpha[:, :1, ...])
            alpha = torch.cat([alpha, ones], dim=1)
            probability = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1, ...]), 1.-alpha+1e-10], dim=1), dim=1)[:, :-1, ...]
            self.outputs["probability"] = probability
            self.outputs["logits"] = torch.cat([self.outputs["logits"], ones], dim=1)
        else:
            self.outputs["probability"] = self.softmax(self.outputs["logits"])
            
        if self.use_mixture_loss:
            sigma = self.sigmoid(self.convs["sigmaconv"](x))
            sigma = torch.clamp(sigma, 0.01, 1.)
            self.outputs["sigma"] = sigma
            self.outputs["pi"] = pi = self.outputs["probability"]
            weights = pi / sigma
            weights = weights * padding_mask
            weights = weights / weights.sum(1, True)
            self.outputs["probability"] = weights
            candidates_idx = weights.argmax(1, True)
            #self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)#
            
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        #print(self.outputs["probability"][0, :, 200, 600].max())
        self.outputs["depth"] = 0.1 * 0.58 * W / self.outputs["disp"]

        return self.outputs
    
    
class DepthDecoderContinuous(nn.Module):
    def __init__(self, num_ch_enc, 
                 no_levels=49, 
                 disp_min=2, 
                 disp_max=300, 
                 num_ep=0,
                 pe_type="neural",
                 use_skips=True, 
                 use_denseaspp=True, 
                 xz_levels=0, 
                 xz_min=0.1852, xz_max=0.3704, #xz_min=0.2315, xz_max=0.3426,#
                 use_mixture_loss=False, 
                 render_probability=False,
                 plane_residual=False):
        super(DepthDecoderContinuous, self).__init__()

        self.no_levels = no_levels
        self.xz_levels = xz_levels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.disp_min = disp_min
        self.disp_max = disp_max
        self.xz_min = xz_min
        self.xz_max = xz_max
        self.num_ep = num_ep
        self.pe_type = pe_type
        self.use_mixture_loss = use_mixture_loss
        self.render_probability = render_probability
        self.plane_residual = plane_residual

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.use_denseaspp = use_denseaspp

        print("use {} xy plane and {} xz plane.".format(self.no_levels, self.xz_levels))
        
        # decoder
        self.convs = OrderedDict()
        
        if self.num_ep > 0:
            if self.pe_type == "neural":
                self.convs["epconv"] = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True),
                nn.Conv2d(16, self.num_ep, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ELU(inplace=True)
                )
            elif self.pe_type == "frequency":
                self.convs["epconv"] = get_embedder((self.num_ep//2 - 1)//2)
        
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1]+self.num_ep if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if i > 0:
                num_ch_in += self.num_ep
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        if use_denseaspp:
            print("use DenseAspp Block")
            self.convs["denseaspp"] = DenseAspp()
            
        self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
            
        if render_probability:
            self.convs["piconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels - 1)
        else:
            self.convs["piconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
        
        if self.use_mixture_loss:
            print("use mixture Lap loss")
            self.convs["sigmaconv"] = Conv3x3(self.num_ch_dec[0], self.no_levels + self.xz_levels)
            
        if self.plane_residual:
            print("use plane residual")
            self.convs["residualconv"] = nn.Sequential(nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], 1),
                                                       nn.AdaptiveAvgPool2d((1, 1)),
                                                       nn.Conv2d(self.num_ch_dec[0], self.no_levels + self.xz_levels, 1))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        

    def forward(self, input_features, input_grids=None):
        self.outputs = {}
        
        if self.num_ep > 0:
            grids_ep = self.convs["epconv"](input_grids)

        # decoder
        x = input_features[-1]
        if self.num_ep > 0:
            dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
            x = torch.cat([x, dgrid], dim=1)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if self.num_ep > 0 and i > 0:
                dgrid = F.interpolate(grids_ep, size=(x.shape[2], x.shape[3]), align_corners=True, mode='bilinear')
                x = torch.cat([x, dgrid], dim=1)
            x = self.convs[("upconv", i, 1)](x)
            
            if i == 4 and self.use_denseaspp:
                x = self.convs["denseaspp"](x)

        B, _, H, W = x.shape

        disp_levels = self.sigmoid(self.convs["dispconv"](x))#B, N, H, W
        self.outputs["disp_levels"] = disp_levels
        disp_layered = self.disp_max * (self.disp_min / self.disp_max)**disp_levels # B, N, H, W

        self.outputs["disp_layered"] = disp_layered
        logits = self.convs["piconv"](x)
        self.outputs["logits"] = logits
        if self.render_probability:
            depth_layered = 0.1 * 0.58 * W / disp_layered
            dists = depth_layered[:, 1:, ...] - depth_layered[:, :-1, ...]
            # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=1)
            camera_plane = create_camera_plane(height=H, width=W)
            dists = dists * torch.linalg.norm(camera_plane, dim=1, keepdim=True)
            self.outputs["dists"] = dists
            alpha = 1. - torch.exp(-F.relu(self.outputs["logits"]) * dists)
            ones = torch.ones_like(alpha[:, :1, ...])
            alpha = torch.cat([alpha, ones], dim=1)
            probability = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1, ...]), 1.-alpha+1e-10], dim=1), dim=1)[:, :-1, ...]
            self.outputs["probability"] = probability
            self.outputs["logits"] = torch.cat([self.outputs["logits"], ones], dim=1)
        else:
            self.outputs["probability"] = self.softmax(self.outputs["logits"])
            
        if self.use_mixture_loss:
            sigma = self.sigmoid(self.convs["sigmaconv"](x))
            sigma = torch.clamp(sigma, 0.01, 1.)
            self.outputs["sigma"] = sigma
            self.outputs["pi"] = pi = self.outputs["probability"]
            weights = pi / sigma
            # weights = weights * padding_mask
            weights = weights / weights.sum(1, True)
            self.outputs["probability"] = weights
            candidates_idx = weights.argmax(1, True)
            #self.outputs["disp"] = torch.gather(self.outputs["disp_layered"], 1, candidates_idx)#
            
        self.outputs["disp"] = (self.outputs["probability"] * self.outputs["disp_layered"]).sum(1, True)
        self.outputs["depth"] = 0.1 * 0.58 * W / self.outputs["disp"]

        return self.outputs
