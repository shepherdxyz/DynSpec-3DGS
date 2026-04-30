#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.sh_utils import eval_sh
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from cubemapencoder import CubemapEncoder
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, positional_encoding, get_pencoding_len
from pytorch3d.transforms import quaternion_apply, quaternion_to_matrix
from pytorch3d.ops import knn_points


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.refl_activation = torch.sigmoid
        self.inverse_refl_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        def reset_neighbors():
            self.knn_to_track = 16
            knns = knn_points(self.get_xyz[None], self.get_xyz[None], K=16)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]
        self.neibor_activation = reset_neighbors


    # init_refl_v: do not need to be set when rendering
    def __init__(self, sh_degree = -1):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree 
        self._xyz = torch.empty(0)
        self._init_xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._refl_strength = torch.empty(0)
        self._features_dc = torch.empty(0) # SH base impl
        self._features_rest = torch.empty(0) # SH base impl
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.free_radius = 0
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_refl_value = 1e-3

        self.env_map = None
        self.deform_env_map = None
        self.setup_functions()
        # KNN information for training regularization
        self.keep_track_of_knn = True

    # 多的函数
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._refl_strength,
            self._features_dc,
            self._features_rest,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    # 多的函数
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._refl_strength,
        self._features_dc,
        self._features_rest,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    # 多的函数
    def set_opacity_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    # 多的函数
    @property
    def get_refl(self):
        return self.refl_activation(self._refl_strength)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    # 多的函数
    @property
    def get_envmap(self): # 
        return self.env_map
    # 多的函数
    @property
    def get_deform_envmap(self): # 
        return self.deform_env_map
    
    # 多的函数
    @property
    def get_refl_strength_to_total(self):
        refl = self.get_refl
        return (refl>0.1).sum() / refl.shape[0]
    # 多的函数
    def get_sh_color(self, cam_o, ret_dir_pp = False):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - cam_o.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        sh_color = torch.clamp_min(sh2rgb + 0.5, 0.0)
        if ret_dir_pp: return sh_color, dir_pp_normalized
        else: return sh_color
    # 多的函数
    def get_depth(self, proj_mat):
        pts = self.get_xyz
        cpts = torch.cat([pts, torch.ones(pts.shape[0], 1).cuda()], dim=-1)
        tpts = (proj_mat @ cpts.T).T
        tpts = tpts[:,:3] #/ tpts[:,3:]
        z = tpts[:, 2:3]#*0.5 + 0.5
        return z
    # 多的函数
    def get_min_axis(self, cam_o):
        pts = self.get_xyz
        p2o = cam_o[None] - pts
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)

        rot_matrix = build_rotation(self.get_rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

        neg_msk = torch.sum(p2o*ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk] # make sure normal orient to camera
        return ndir 
    # 多的函数 取最短轴作为法线方向
    def get_min_axis_with_t(self, cam_o, d_xyz, d_rotation, d_scaling):
        pts = self.get_xyz + d_xyz
        p2o = cam_o[None] - pts
        scales = self.get_scaling + d_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)

        rot_matrix = build_rotation(self.get_rotation + d_rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

        neg_msk = torch.sum(p2o*ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk] # make sure normal orient to camera 
        return ndir # 法线方向 单位向量
    #def get_covariance(self, scaling_modifier = 1):
    #    return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        #self.max_sh_degree = 0
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    # 多的函数
    def init_properties_from_pcd(self, pts, colors):
        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(pts), 0.0000001)
        self.free_radius = torch.sqrt(dist2.max())
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # KNN--Find the distance of the closest point to determine the initial scale (avoid holes)
        rots = torch.zeros((pts.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))
        refl = self.inverse_refl_activation(torch.ones_like(opacities).cuda() * self.init_refl_value) ##
        return {
            'opac': opacities, 'rot':rots, 'scale':scales, 'shs':features, 'refl':refl
        }

    def create_from_pcd(self, pcd, spatial_lr_scale: float, cubemap_resol = 256):
        self.spatial_lr_scale = spatial_lr_scale
        # 改动很大
        pts = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        base_prop = self.init_properties_from_pcd(pts, colors)
        base_prop['xyz'] = pts
        print("Number of base points at initialisation : ", pts.shape[0])

        for key in base_prop.keys():
            base_prop[key] = base_prop[key].cuda()
        tot_props = base_prop

        self._xyz = nn.Parameter(tot_props['xyz'].requires_grad_(True))
        self._scaling = nn.Parameter(tot_props['scale'].requires_grad_(True))
        self._rotation = nn.Parameter(tot_props['rot'].requires_grad_(True))
        self._opacity = nn.Parameter(tot_props['opac'].requires_grad_(True))
        self._features_dc = nn.Parameter(tot_props['shs'][:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(tot_props['shs'][:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        
        self._refl_strength = nn.Parameter(tot_props['refl'].requires_grad_(True))
        # env_map = CubemapEncoder(output_dim=3, resolution=cubemap_resol)
        env_map = CubemapEncoder(output_dim=3, resolution=cubemap_resol)
        self.env_map = env_map.cuda()
        deform_env_map = CubemapEncoder(output_dim=3, resolution=cubemap_resol)
        self.deform_env_map = deform_env_map.cuda()


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 多了 self._refl_strength ， self.env_map.parameters()
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._refl_strength], 'lr': training_args.refl_lr, "name": "refl"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},
            {'params': self.deform_env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "deform_env"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        ###
        #self.optimizer.add_param_group({'params': self.mlp.parameters(), 'lr': training_args.mlp_lr, "name": "mlp"})
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    def update_cubemap_learning_rate(self):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "env":
                lr = 0.1
                param_group['lr'] = lr
                return lr
    # 多了refl
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('refl')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    # 多了refls
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        refls = self._refl_strength.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, refls, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        ###
        #torch.save(self.mlp.state_dict(), path.replace('.ply', '.ckpt'))
        # 多的  这里保存 env_map文件
        if self.env_map is not None:
            save_path = os.path.dirname(path)
            torch.save(self.env_map.state_dict(), os.path.join(save_path, "env.map"))
            torch.save(self.deform_env_map.state_dict(), os.path.join(save_path, "deformenv.map"))

        # if self.env_map is not None:
        #     save_path = path.replace('.ply', '.map')
        #     torch.save(self.env_map.state_dict(), save_path)  
    # 多的函数 不透明度小于0.01的不变，不透明度大于0.01的全部重置为0.01
    def reset_opacity0(self, resetv):
        RESET_V = 0.01
        #REFL_MSK_THR = 0.1
        #refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        opacity_old = self.get_opacity
        o_msk = (opacity_old < RESET_V).flatten() # 不透明度小于 RESET_V的数量
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([resetv]).cuda()) # 全部都变成 -4.5951
        opacities_new[o_msk] = self._opacity[o_msk]
        # only reset non-refl gaussians
        #opacities_new[refl_msk] = self._opacity[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity0_v2(self, resetv, exclusive_msk = None):
        RESET_V = 0.01
        #REFL_MSK_THR = 0.1
        #refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        opacity_old = self.get_opacity
        o_msk = (opacity_old < RESET_V).flatten() # 不透明度小于 RESET_V的数量
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([resetv]).cuda()) # 全部都变成 -4.5951
        opacities_new[o_msk] = self._opacity[o_msk]
        opacities_new[~exclusive_msk] = inverse_sigmoid(torch.tensor([0.2]).cuda())
        # only reset non-refl gaussians
        #opacities_new[refl_msk] = self._opacity[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]
    # 多的函数，不透明度大于0.9的不变，不透明度小于0.9的全部重置为0.9
    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        #REFL_MSK_THR = 0.1
        #refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        # only reset refl gaussians
        #opacities_new[refl_msk] = self._opacity[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]
    # 多的函数，没看到调用
    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        #REFL_MSK_THR = 0.1
        #refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        # only reset refl gaussians
        #opacities_new[refl_msk] = self._opacity[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]
    # 多的函数
    def reset_refl(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = self._refl_strength[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl")
        if "refl" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl"]
    def reset_xyz(self, d_xyz, exclusive_msk = None):
        xyz_new = self.get_xyz
        if exclusive_msk is not None:
            xyz_new[exclusive_msk] = self._xyz[exclusive_msk] + d_xyz[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        if "xyz" not in optimizable_tensors: return
        self._xyz = optimizable_tensors["xyz"]
    # 多的函数  自己改的
    def reset_refl_specify(self, exclusive_msk = None):
        refl_new = inverse_sigmoid(torch.max(self.get_refl, torch.ones_like(self.get_refl)*self.init_refl_value))
        if exclusive_msk is not None:
            refl_new[exclusive_msk] = 0.85
        optimizable_tensors = self.replace_tensor_to_optimizer(refl_new, "refl")
        if "refl" not in optimizable_tensors: return
        self._refl_strength = optimizable_tensors["refl"]
    # 多的函数，没看到调用
    def dist_rot(self): #
        REFL_MSK_THR = 0.1
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[refl_msk] = rot[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]
    # 多的函数，重置不透明度到很大会用到
    def dist_color(self, exclusive_msk = None):
        REFL_MSK_THR = 0.05
        DIST_RANGE = 0.4
        refl_msk = self.get_refl.flatten() > REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE) # ~0.4~0.4
        dist_dcc[refl_msk] = dcc[refl_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]
    # 多的函数，重置不透明度到很大会用到
    def enlarge_refl_scales(self, ret_raw = True, ENLARGE_SCALE=1.5, REFL_MSK_THR = 0.02, exclusive_msk = None):
        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        rmin_axis = (torch.ones_like(scales)*ENLARGE_SCALE).scatter(1, min_axis_id, 1)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales*rmin_axis)
            # only reset refl gaussians
            scale_new[refl_msk] = self._scaling[refl_msk]
        else:
            scale_new = scales*rmin_axis
            scale_new[refl_msk] = scales[refl_msk]
        return scale_new
    # 多的函数 每看到哪里用到了
    def enlarge_refl_scales_strategy2(self, ret_raw = True, ENLARGE_SCALE=1.36, REFL_MSK_THR = 0.02, exclusive_msk = None):
        refl_msk = self.get_refl.flatten() < REFL_MSK_THR
        if exclusive_msk is not None:
            refl_msk = torch.logical_or(refl_msk, exclusive_msk)
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        rmin_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1) #001
        rmax2_axis = 1 - rmin_axis #110
        smax = torch.max(scales, dim=-1, keepdim=True).values.expand_as(scales)
        scale_new = smax*rmax2_axis*ENLARGE_SCALE+scales*rmin_axis
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scale_new)
            # only reset refl gaussians
            scale_new[refl_msk] = self._scaling[refl_msk]
        else:
            scale_new[refl_msk] = scales[refl_msk]
        return scale_new
    # 多的函数 重置不透明度到很大用到了 已经被注释掉了
    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_refl_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        # 多了一行
        refls = np.asarray(plydata.elements[0]["refl"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # 多了一行这个
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        # 多了一行这个
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # 多了一行这个
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        #mlp_path = path.replace('.ply', '.ckpt')
        #if os.path.exists(mlp_path):
        #    self.mlp.load_state_dict(torch.load(mlp_path))
        # 多的
        # map_path = path.replace('.ply', '.map')
        map_path = os.path.dirname(path)
        if os.path.exists(map_path):
            self.env_map = CubemapEncoder(output_dim=3, resolution=128).cuda()
            self.env_map.load_state_dict(torch.load(os.path.join(map_path, "env.map")))
            self.deform_env_map = CubemapEncoder(output_dim=3, resolution=128).cuda()
            self.deform_env_map.load_state_dict(torch.load(os.path.join(map_path, "deformenv.map")))
        # 多了 _refl_strength
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._refl_strength = nn.Parameter(torch.tensor(refls, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    # 多了 if group["name"] == "mlp" or group["name"] == "env": continue
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "deform_env": continue 
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    # 多了一个 refl
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._refl_strength = optimizable_tensors['refl']
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    # if group["name"] == "mlp" or group["name"] == "env": continue 什么意思
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "deform_env": continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    # 多了new_refl,self._features_dc,self._features_rest
    def densification_postfix(self, new_xyz, new_refl, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "refl": new_refl,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._refl_strength = optimizable_tensors['refl']
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    # 多了一个 new_refl
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_refl = self._refl_strength[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_refl, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    # 多了一个 new_refl
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_refl = self._refl_strength[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_refl, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    # 一样的
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_with_return(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return prune_mask
    # 一样的
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None, 
                                   deformed_xyz=None, deformed_scaling=None, deformed_rotation=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True,):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        # scaling = self.get_scaling # (N,3)
        scaling = deformed_scaling[mask] # mask (N), scaling (N-,3), deformed_scaling(N,3)

        
        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0]) # (N-)
        
        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs() # (N-)
        # cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)
        cum_probs = areas / areas.sum(dim=-1, keepdim=True) # (N-)
        
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True) # (N_sample)
        if mask is not None:
            valid_indices = torch.arange(len(deformed_xyz), device="cuda")[mask] # (N-)  N-是radii大于0的高斯的数量
            random_indices = valid_indices[random_indices] # 这行代码有问题 # (N_sample)
        
        random_points = deformed_xyz[random_indices] + quaternion_apply( # random_points,deformed_xyz[random_indices] (N_sample,3)
            deformed_rotation[random_indices],  # (N_sample,4)
            sampling_scale_factor * deformed_scaling[random_indices] * torch.randn_like(deformed_xyz[random_indices])) #deformed_scaling[random_indices],torch.randn_like(deformed_xyz[random_indices]) (N_sample,3)
        
        return random_points, random_indices # (N_sample,3),(N_sample)
    
    def get_field_values(self, x, gaussian_idx=None, 
                    closest_gaussians_idx=None,
                    gaussian_strengths=None, 
                    gaussian_centers=None, 
                    gaussian_rotation=None,
                    gaussian_scaling=None,
                    gaussian_inv_scaled_rotation=None,
                    return_sdf=True, density_threshold=1., density_factor=1.,
                    return_sdf_grad=False, sdf_grad_max_value=10.,
                    opacity_min_clamp=1e-16,
                    return_closest_gaussian_opacities=False,
                    return_beta=False,):

        gaussian_strengths = self.get_opacity # (N,1)
        # gaussian_centers = self.get_xyz
        gaussian_inv_scaled_rotation = self.get_covariance(gaussian_rotation, gaussian_scaling, return_full_matrix=True, return_sqrt=True, inverse_scales=True) # 这里倒数了缩放矩阵 # (N,3,3)
        
        if closest_gaussians_idx is None:
            closest_gaussians_idx = self.knn_idx[gaussian_idx] # (N_sample,16)
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx] # (N_sample,16,3)
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx] # (N_sample,16,3,3)
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx] # (N_sample,16,1)
        
        fields = {}
        
        # Compute the density field as a sum of local gaussian opacities
        # TODO: Change the normalization of the density (maybe learn the scaling parameter?)
        shift = (x[:, None] - closest_gaussian_centers) # (N_sample,16,3)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None] # (N_sample,16,3,1)
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8) # (N_sample,16)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)  # (N_sample,16)
        densities = neighbor_opacities.sum(dim=-1)  # (N_sample)
        fields['density'] = densities.clone()  # (N_sample)
        density_mask = densities >= 1. # (N_sample)
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)
        
        if return_closest_gaussian_opacities:
            fields['closest_gaussian_opacities'] = neighbor_opacities  # (N_sample,16)
        
        if return_sdf or return_sdf_grad or return_beta:
            # --- Old way
            # beta = self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
            # ---New way
            beta = self.get_beta(x,  # (N_sample)
                                 closest_gaussians_idx=closest_gaussians_idx, 
                                 closest_gaussians_opacities=neighbor_opacities, 
                                 densities=densities,
                                 opacity_min_clamp=opacity_min_clamp,
                                 )
            clamped_densities = densities.clamp(min=opacity_min_clamp) # (N_sample)

        if return_beta:
            fields['beta'] = beta # (N_sample)
        
        # Compute the signed distance field
        if return_sdf:
            sdf_values = beta * (
                torch.sqrt(-2. * torch.log(clamped_densities)) # TODO: Change the max=1. to something else?
                - np.sqrt(-2. * np.log(min(density_threshold, 1.)))
                )
            fields['sdf'] = sdf_values
            
        # Compute the gradient of the signed distance field
        if return_sdf_grad:
            sdf_grad = neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]
            sdf_grad = sdf_grad.sum(dim=-2)
            sdf_grad = (beta / (clamped_densities * torch.sqrt(-2. * torch.log(clamped_densities))).clamp(min=opacity_min_clamp))[..., None] * sdf_grad
            fields['sdf_grad'] = sdf_grad.clamp(min=-sdf_grad_max_value, max=sdf_grad_max_value)
            
        return fields
    
    def get_covariance(self, gaussian_rotation, gaussian_scaling, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        scaling = gaussian_scaling
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = quaternion_to_matrix(gaussian_rotation) * scaling[:, None]
        if return_sqrt:
            return scaled_rotation
        
        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix
        
        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device="cuda")
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
        
        return cov3D
    
    def get_beta(self, x, 
                 closest_gaussians_idx=None, 
                 closest_gaussians_opacities=None,
                 densities=None,
                 opacity_min_clamp=1e-32,):
        """_summary_

        Args:
            x (_type_): Should have shape (n_points, 3)
            closest_gaussians_idx (_type_, optional): Should have shape (n_points, n_neighbors).
                Defaults to None.
            closest_gaussians_opacities (_type_, optional): Should have shape (n_points, n_neighbors).
            densities (_type_, optional): Should have shape (n_points, ).

        Returns:
            _type_: _description_
        """
        # if self.beta_mode == 'learnable':
        #     return torch.exp(self._log_beta).expand(len(x))
        
        # elif self.beta_mode == 'average':
        if closest_gaussians_idx is None:
            raise ValueError("closest_gaussians_idx must be provided when using beta_mode='average'.")
        return self.get_scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
        
        # elif self.beta_mode == 'weighted_average':
        #     if closest_gaussians_idx is None:
        #         raise ValueError("closest_gaussians_idx must be provided when using beta_mode='weighted_average'.")
        #     if closest_gaussians_opacities is None:
        #         raise ValueError("closest_gaussians_opacities must be provided when using beta_mode='weighted_average'.")
            
        #     min_scaling = self.scaling.min(dim=-1)[0][closest_gaussians_idx]
            
        #     # if densities is None:
        #     if True:
        #         opacities_sum = closest_gaussians_opacities.sum(dim=-1, keepdim=True)
        #     else:
        #         opacities_sum = densities.view(-1, 1)
        #     # weights = neighbor_opacities.clamp(min=opacity_min_clamp) / opacities_sum.clamp(min=opacity_min_clamp)
        #     weights = closest_gaussians_opacities / opacities_sum.clamp(min=opacity_min_clamp)

        #     # Three methods to handle the case where all opacities are 0.
        #     # Important because we need to avoid beta == 0 at all cost for these points!
        #     # Indeed, beta == 0. gives sdf == 0.
        #     # However these points are far from gaussians, so they should have a sdf != 0.

        #     # Method 1: Give 1-weight to closest gaussian (Not good)
        #     if False:
        #         one_at_closest_gaussian = torch.zeros(1, neighbor_opacities.shape[1], device=rc.device)
        #         one_at_closest_gaussian[0, 0] = 1.
        #         weights[opacities_sum[..., 0] == 0.] = one_at_closest_gaussian
        #         beta = (rc.scaling.min(dim=-1)[0][closest_gaussians_idx] * weights).sum(dim=1)
            
        #     # Method 2: Give the maximum scaling value in neighbors as beta (Not good if neighbors have low scaling)
        #     if False:
        #         beta = (min_scaling * weights).sum(dim=-1)
        #         mask = opacities_sum[..., 0] == 0.
        #         beta[mask] = min_scaling.max(dim=-1)[0][mask]
            
        #     # Method 3: Give a constant, large beta value (better control)
        #     if True:
        #         beta = (min_scaling * weights).sum(dim=-1)
        #         with torch.no_grad():
        #             if False:
        #                 # Option 1: beta = camera_spatial_extent
        #                 beta[opacities_sum[..., 0] == 0.] = rc.get_cameras_spatial_extent()
        #             else:
        #                 # Option 2: beta = largest min_scale in the scene
        #                 beta[opacities_sum[..., 0] == 0.] = min_scaling.max().detach()
            
        #     return beta
        
        # else:
        #     raise ValueError("Unknown beta_mode.")

    def reset_neighbors(self, knn_to_track:int=None):

        knn_to_track = 16
        # Compute KNN               
        with torch.no_grad():
            self.knn_to_track = knn_to_track
            knns = knn_points(self.get_xyz[None], self.get_xyz[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]