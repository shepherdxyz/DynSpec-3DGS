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
import math, time
import torch.nn.functional as F
import diff_gaussian_rasterization_c3
import diff_gaussian_rasterization_c7 
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import numpy as np

# rayd: x,3, from camera to world points
# normal: x,3
# all normalized
# 计算一条射线在给定法线上的反射方向
def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl
# 根据反射方向 rays_d 查询环境贴图 envmap，返回每个像素的反射颜色。
def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)

def get_refl_color_with_t(envmap: torch.Tensor, HWK, R, T, normal_map, env_deform, time_input): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    rays_d = env_deform.step(rays_d.reshape(-1,3), time_input)
    rays_d = rays_d.reshape(HWK[0],HWK[1],3)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)

def get_refl_color_with_dynamiccube(envmap: torch.Tensor, HWK, R, T, normal_map, env_deform, time_input): #RT W2C
    # rays_d = sample_camera_rays(HWK, R, T)
    # rays_d = reflection(rays_d, normal_map)
    x_coords = torch.arange(HWK[1]).repeat(HWK[0], 1)  # 每一行的 x 坐标
    y_coords = torch.arange(HWK[0]).view(-1, 1).repeat(1, HWK[1])  # 每一列的 y 坐标
    coords = torch.stack((x_coords, y_coords), dim=-1).view(-1, 2).cuda()
    deformed_color = envmap(env_deform.step(coords, time_input))
    deformed_color = torch.sigmoid(deformed_color)
    # rays_d = env_deform.step(rays_d.reshape(-1,3), time_input)
    # rays_d = rays_d.reshape(HWK[0],HWK[1],3)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return deformed_color

def get_refl_color_with_dynamiccube_normal(envmap: torch.Tensor, HWK, R, T, normal_map, env_deform, time_input): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    # x_coords = torch.arange(HWK[1]).repeat(HWK[0], 1)  # 每一行的 x 坐标
    # y_coords = torch.arange(HWK[0]).view(-1, 1).repeat(1, HWK[1])  # 每一列的 y 坐标
    # coords = torch.stack((x_coords, y_coords), dim=-1).view(-1, 2).cuda()
    deformed_color = envmap(env_deform.step(rays_d.reshape(-1,3), time_input))
    deformed_color = torch.sigmoid(deformed_color)
    # rays_d = env_deform.step(rays_d.reshape(-1,3), time_input)
    # rays_d = rays_d.reshape(HWK[0],HWK[1],3)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return deformed_color

def render_env_map(pc: GaussianModel):
    env_cood1 = sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap)
    env_cood2 = sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap)
    return {'env_cood1': env_cood1, 'env_cood2': env_cood2}

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, d_xyz, d_rotation, d_scaling, env_deform, time_input, scaling_modifier = 1.0, initial_stage = False, deform_envmap = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation
    shs = pc.get_features
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    normals = pc.get_min_axis_with_t(viewpoint_camera.camera_center, d_xyz, d_rotation, d_scaling) # x,3
    refl_ratio = pc.get_refl  # x,1

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] # 1,H,W 这里返回的是什么？是每个像素的反射强度吗？
    normal_map = out_ts[3:6,...] # 3,H,W

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    if deform_envmap: # 这里写错了，上下要改一下  默认是true的话，渲染的时候怎么办  还没改好
        refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    else:
        refl_color = get_refl_color_with_t(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, env_deform, time_input)
    # 这里的 pc.get_envmap要不要加入时间信息
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii,  # 这个 _radii 貌似 是2D高斯（在屏幕空间近似成圆）在屏幕空间以均值为中心覆盖范围的半径（单位为像素）
        "noarmal": normals
    }
        
    return results

def render_dynamiccube(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, d_xyz, d_rotation, d_scaling, env_deform, time_input, scaling_modifier = 1.0, initial_stage = False, deform_envmap = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation
    shs = pc.get_features
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    normals = pc.get_min_axis_with_t(viewpoint_camera.camera_center, d_xyz, d_rotation, d_scaling) # x,3
    refl_ratio = pc.get_refl  # x,1

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] # 1,H,W 这里返回的是什么？是每个像素的反射强度吗？
    normal_map = out_ts[3:6,...] # 3,H,W

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    if deform_envmap: # 这里写错了，上下要改一下  默认是true的话，渲染的时候怎么办  还没改好
        refl_color = refl_color
    else:
        dynamic_cube_color = get_refl_color_with_dynamiccube(pc.get_deform_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, env_deform, time_input)
        refl_color = refl_color + dynamic_cube_color.view(270, 480, 3).permute(2, 0, 1)
    # 这里的 pc.get_envmap要不要加入时间信息
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii,  # 这个 _radii 貌似 是2D高斯（在屏幕空间近似成圆）在屏幕空间以均值为中心覆盖范围的半径（单位为像素）
        "noarmal": normals
    }
        
    return results

def render_dynamiccube_normal(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, d_xyz, d_rotation, d_scaling, env_deform, time_input, scaling_modifier = 1.0, initial_stage = False, deform_envmap = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation
    shs = pc.get_features
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    normals = pc.get_min_axis_with_t(viewpoint_camera.camera_center, d_xyz, d_rotation, d_scaling) # x,3
    refl_ratio = pc.get_refl  # x,1

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] # 1,H,W 这里返回的是什么？是每个像素的反射强度吗？
    normal_map = out_ts[3:6,...] # 3,H,W

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    if deform_envmap: # 这里写错了，上下要改一下  默认是true的话，渲染的时候怎么办  还没改好
        refl_color = refl_color
    else:
        dynamic_cube_color = get_refl_color_with_dynamiccube_normal(pc.get_deform_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, env_deform, time_input)
        refl_color = refl_color + dynamic_cube_color.view(270, 480, 3).permute(2, 0, 1)
    # 这里的 pc.get_envmap要不要加入时间信息
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii,  # 这个 _radii 貌似 是2D高斯（在屏幕空间近似成圆）在屏幕空间以均值为中心覆盖范围的半径（单位为像素）
        "noarmal": normals
    }
        
    return results

def render_dynamiccube_normal_static(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, d_xyz, d_rotation, d_scaling, env_deform, time_input, scaling_modifier = 1.0, initial_stage = False, deform_envmap = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation
    shs = pc.get_features

    means3D_static = pc.get_xyz
    scales_static = pc.get_scaling
    rotations_static = pc.get_rotation
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    normals = pc.get_min_axis_with_t(viewpoint_camera.camera_center, d_xyz, d_rotation, d_scaling) # x,3
    normals_static = pc.get_min_axis(viewpoint_camera.camera_center)
    refl_ratio = pc.get_refl  # x,1

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    input_ts_static = torch.cat([torch.zeros_like(normals_static), normals_static, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)

    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    out_ts_static, _radii_static = rasterizer_c7(
        means3D = means3D_static,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts_static,
        opacities = opacities,
        scales = scales_static,
        rotations = rotations_static,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] # 1,H,W 这里返回的是什么？是每个像素的反射强度吗？
    normal_map = out_ts[3:6,...] # 3,H,W

    normal_map_static = out_ts_static[3:6,...]
    normal_map_static = normal_map_static.permute(1,2,0)
    normal_map_static = normal_map_static / (torch.norm(normal_map_static, dim=-1, keepdim=True)+1e-6)

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map_static)
    if deform_envmap: # 这里写错了，上下要改一下  默认是true的话，渲染的时候怎么办  还没改好
        refl_color = refl_color
    else:
        dynamic_cube_color = get_refl_color_with_dynamiccube_normal(pc.get_deform_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, env_deform, time_input)
        refl_color = refl_color + dynamic_cube_color.view(270, 480, 3).permute(2, 0, 1)
    # 这里的 pc.get_envmap要不要加入时间信息
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii,  # 这个 _radii 貌似 是2D高斯（在屏幕空间近似成圆）在屏幕空间以均值为中心覆盖范围的半径（单位为像素）
        "noarmal": normals
    }
        
    return results

def render_dynamiccube_normal_staticv2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, d_xyz, d_rotation, d_scaling, env_deform, time_input, scaling_modifier = 1.0, initial_stage = False, deform_envmap = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation
    shs = pc.get_features

    means3D_static = pc.get_xyz
    scales_static = pc.get_scaling
    rotations_static = pc.get_rotation
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    # normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    normals = pc.get_min_axis_with_t(viewpoint_camera.camera_center, d_xyz, d_rotation, d_scaling) # x,3
    normals_static = pc.get_min_axis(viewpoint_camera.camera_center)
    refl_ratio = pc.get_refl  # x,1

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    input_ts_static = torch.cat([torch.zeros_like(normals_static), normals_static, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)

    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    out_ts_static, _radii_static = rasterizer_c7(
        means3D = means3D_static,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts_static,
        opacities = opacities,
        scales = scales_static,
        rotations = rotations_static,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] # 1,H,W 这里返回的是什么？是每个像素的反射强度吗？
    normal_map = out_ts[3:6,...] # 3,H,W

    normal_map_static = out_ts_static[3:6,...]
    normal_map_static = normal_map_static.permute(1,2,0)
    normal_map_static = normal_map_static / (torch.norm(normal_map_static, dim=-1, keepdim=True)+1e-6)
    refl_strength_static = out_ts_static[6:7,...]
    base_color_static = out_ts_static[:3,...]

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map_static)
    if deform_envmap: # 这里写错了，上下要改一下  默认是true的话，渲染的时候怎么办  还没改好
        refl_color = refl_color
        final_image = (1-refl_strength_static) * base_color_static + refl_strength_static * refl_color
    else:
        dynamic_cube_color = get_refl_color_with_dynamiccube_normal(pc.get_deform_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, env_deform, time_input)
        refl_color = refl_color + refl_strength * dynamic_cube_color.view(270, 480, 3).permute(2, 0, 1)
        final_image = (refl_strength_static+refl_strength)/2 * base_color + (refl_strength_static+refl_strength)/2 * refl_color
    # 这里的 pc.get_envmap要不要加入时间信息
    

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii,  # 这个 _radii 貌似 是2D高斯（在屏幕空间近似成圆）在屏幕空间以均值为中心覆盖范围的半径（单位为像素）
        "noarmal": normals
    }
        
    return results