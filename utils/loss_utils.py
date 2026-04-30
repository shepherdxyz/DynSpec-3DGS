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
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
from kornia.filters import bilateral_blur
from math import exp
import lpips
from pytorch3d.ops.knn import knn_points
import math

# 初始化 LPIPS loss，只初始化一次
lpips_loss = lpips.LPIPS(net='vgg').cuda()
lpips_loss.eval()  # 设置为 eval 模式，避免模型参数更新

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# im: 3,H,W, with grad
def bilateral_smooth_img_loss(im: torch.Tensor):
    REFL_THRESH = 0.05
    im = im.mean(dim = 0) # H,W
    msk = im > REFL_THRESH
    if not torch.any(msk): return 0
    cim = im.detach().clone()
    cim[~msk] = -999999.0 # make non refl area huge different with the refl area, so that the bilateral_blur won't blur pixel out of the refl area boundary
    smoothed_im = bilateral_blur(cim[None,None], (11,11), 75/255, (10,10))[0,0]
    loss = l2_loss(im[msk], smoothed_im[msk])
    return loss

# im: 3,H,W, with grad
gBlur = None
def smooth_img_loss(im: torch.Tensor):
    global gBlur
    if gBlur is None:
        gBlur = GaussianBlur(9, 4.0).cuda()
    im_smooth = gBlur(im[None].detach())[0]
    loss = l2_loss(im, im_smooth)
    return loss

def calculate_lpips_loss(img1, img2):
    image_normalized1 = img1 * 2.0 - 1.0
    image_normalized2 = img2 * 2.0 - 1.0
    loss = lpips_loss(image_normalized1.unsqueeze(0), image_normalized2.unsqueeze(0))
    return loss

def reg_dxyz(d_xyz):
    # 对 x 维度的处理
    x = d_xyz[:, 0]

    # 提取正的值和负的值
    x_pos = x[x > 0]
    x_neg = x[x < 0]

    # 计算正值和负值的平均值
    a_x_pos = x_pos.mean() if len(x_pos) > 0 else 0  # 避免无正数时报错
    a_x_neg = x_neg.mean() if len(x_neg) > 0 else 0  # 避免无负数时报错

    # 计算正值和负值部分的L2范数
    l2_norm_x_pos = torch.sqrt((x_pos - a_x_pos) ** 2)
    l2_norm_x_neg = torch.sqrt((x_neg - a_x_neg) ** 2)

    # 对 y 维度的处理
    y = d_xyz[:, 1]
    y_pos = y[y > 0]
    y_neg = y[y < 0]
    a_y_pos = y_pos.mean() if len(y_pos) > 0 else 0
    a_y_neg = y_neg.mean() if len(y_neg) > 0 else 0
    l2_norm_y_pos = torch.sqrt((y_pos - a_y_pos) ** 2)
    l2_norm_y_neg = torch.sqrt((y_neg - a_y_neg) ** 2)

    # 对 z 维度的处理
    z = d_xyz[:, 2]
    z_pos = z[z > 0]
    z_neg = z[z < 0]
    a_z_pos = z_pos.mean() if len(z_pos) > 0 else 0
    a_z_neg = z_neg.mean() if len(z_neg) > 0 else 0
    l2_norm_z_pos = torch.sqrt((z_pos - a_z_pos) ** 2)
    l2_norm_z_neg = torch.sqrt((z_neg - a_z_neg) ** 2)
    dxyz_loss = l2_norm_x_pos.sum() + l2_norm_x_neg.sum() + l2_norm_y_pos.sum() + l2_norm_y_neg.sum() + l2_norm_z_pos.sum() + l2_norm_z_neg.sum()
    return dxyz_loss

def clustering_loss(gaussians, clustering_weight=1.0):

    if gaussians.numel() == 0:
        return torch.tensor(0.0)
    # 计算这些点的均值位置，作为聚集中心
    cluster_center = gaussians.mean(dim=0)
    
    # 计算每个点到聚集中心的距离
    distances = torch.norm(gaussians - cluster_center, dim=-1)
    
    # 计算损失为距离的平均值，鼓励这些点靠近聚集中心
    loss = clustering_weight * distances.mean()
    return loss

def aiap_loss(x_canonical, x_deformed, n_neighbors=5):
    """
    Computes the as-isometric-as-possible loss between two sets of points, which measures the discrepancy
    between their pairwise distances.

    Parameters
    ----------
    x_canonical : array-like, shape (n_points, n_dims)
        The canonical (reference) point set, where `n_points` is the number of points
        and `n_dims` is the number of dimensions.
    x_deformed : array-like, shape (n_points, n_dims)
        The deformed (transformed) point set, which should have the same shape as `x_canonical`.
    n_neighbors : int, optional
        The number of nearest neighbors to use for computing pairwise distances.
        Default is 5.

    Returns
    -------
    loss : float
        The AIAP loss between `x_canonical` and `x_deformed`, computed as the L1 norm
        of the difference between their pairwise distances. The loss is a scalar value.
    Raises
    ------
    ValueError
        If `x_canonical` and `x_deformed` have different shapes.
    """

    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                             x_canonical.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)

    dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
    dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

    loss = l1_loss(dists_canonical, dists_deformed)

    return loss

def aiap_lossv2(x_canonical, x_deformed, n_neighbors=5): # 这个v2是多加了一个平均
    """
    Computes the as-isometric-as-possible loss between two sets of points, which measures the discrepancy
    between their pairwise distances.

    Parameters
    ----------
    x_canonical : array-like, shape (n_points, n_dims)
        The canonical (reference) point set, where `n_points` is the number of points
        and `n_dims` is the number of dimensions.
    x_deformed : array-like, shape (n_points, n_dims)
        The deformed (transformed) point set, which should have the same shape as `x_canonical`.
    n_neighbors : int, optional
        The number of nearest neighbors to use for computing pairwise distances.
        Default is 5.

    Returns
    -------
    loss : float
        The AIAP loss between `x_canonical` and `x_deformed`, computed as the L1 norm
        of the difference between their pairwise distances. The loss is a scalar value.
    Raises
    ------
    ValueError
        If `x_canonical` and `x_deformed` have different shapes.
    """

    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                             x_canonical.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)

    dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
    dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

    loss = l1_loss(dists_canonical, dists_deformed)/x_canonical.shape[0]

    return loss

def non_orthogonal_loss(v1, v2, lambda_weight=1.0, eps=1e-8):
    # 点积
    dot_product = torch.sum(v1 * v2, dim=-1)
    # 损失项：远离零
    loss = lambda_weight / (dot_product ** 2 + eps)
    return loss.mean()

def sigmoid_ramp(step, start_steps, total_steps, start, end, k=10):
    """平滑上升函数 (Sigmoid)，支持自定义初始值"""
    center = (total_steps - start_steps) / 2  # Sigmoid中心位置
    sigmoid = 1 / (1 + math.exp(-k * (step - center) / total_steps))
    
    # 缩放和平移，确保初始值为 start，结束值为 end
    coef = start + (end - start) * sigmoid
    return coef