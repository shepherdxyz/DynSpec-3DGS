
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, calculate_lpips_loss, reg_dxyz, aiap_lossv2
from gaussian_renderer import render_dynamiccube_normal, render_env_map
import sys
from scene import Scene, GaussianModel, DeformModel, Env_DeformModel,Cube_DeformModel,Cube_DeformModel_Normal
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import cv2, time
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from utils.image_utils import psnr
import torchvision
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

'''
Add decay opacity for refl gaussians (banned)
Add reset refl ratio
Add refl smooth loss (banned)
SH0, and no densify (banned)
INIT_ITER 5000, cbmp_lr = 0.01 $

densify -> 30k
densify_intv in prop -> 100
'''

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    INIT_UNITIL_ITER = opt.init_until_iter #3000
    START_DEFORM_ENCMAP = opt.start_deform_env
    FR_OPTIM_FROM_ITER = opt.feature_rest_from_iter
    NORMAL_PROP_UNTIL_ITER = opt.normal_prop_until_iter + opt.longer_prop_iter #normal_prop_until_iter 默认24_000
    OPAC_LR0_INTERVAL = opt.opac_lr0_interval # 200
    DENSIFIDATION_INTERVAL_WHEN_PROP = opt.densification_interval_when_prop #500
    
    TOT_ITER = opt.iterations + opt.longer_prop_iter + 1
    DENSIFY_UNTIL_ITER = opt.densify_until_iter
    DENSIFY_UNTIL_ITER_NORAML = opt.densify_until_iter + opt.longer_prop_iter

    # for real scenes
    USE_ENV_SCOPE = opt.use_env_scope # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    gaussians = GaussianModel(dataset.sh_degree)

    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    env_deform = Cube_DeformModel_Normal()
    deform.train_setting(opt) # 还没检查设置相应参数
    env_deform.train_setting(opt)
    print(deform.deform)
    print(env_deform.cube_deform)


    scene = Scene(dataset, gaussians) # init all parameters(pos,scale,rot...) from pcds
    gaussians.training_setup(opt)
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    best_psnr = 0.0
    best_iteration = 0
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print('propagation until: {}'.format(NORMAL_PROP_UNTIL_ITER))
    print('densify until: {}'.format(DENSIFY_UNTIL_ITER))
    print('total iter: {}'.format(TOT_ITER))

    initial_stage = True
    deform_envmap = True

    # Toycar
    #ENV_CENTER = torch.tensor([0.6810, 0.8080, 4.4550], device='cuda') # None
    #ENV_RANGE = 2.707

    # Garden
    #ENV_CENTER = torch.tensor([-0.2270,  1.9700,  1.7740], device='cuda') # None
    #ENV_RANGE = 0.974

    # Sedan
    #ENV_CENTER = torch.tensor([-0.032,0.808,0.751], device='cuda') # None
    #ENV_RANGE = 2.138

    while iteration < TOT_ITER:        

        iter_start.record()


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration > FR_OPTIM_FROM_ITER and iteration % 1000 == 0: # 跟deformgs不一样，从一开始就每1000T增加球谐系数
            gaussians.oneupSHdegree()
        if iteration > INIT_UNITIL_ITER:
            initial_stage = False
        if iteration > START_DEFORM_ENCMAP:
            deform_envmap = False
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # if dataset.load2gpu_on_the_fly:
        #     viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        env_time_input = fid.unsqueeze(0).expand(viewpoint_cam.HWK[0] * viewpoint_cam.HWK[1], -1)

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            # ast_noise = 0 if dataset.use_ast else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            if dataset.use_ast:
                ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            else:
                ast_noise = 0
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render_dynamiccube_normal(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, env_deform, env_time_input, initial_stage=initial_stage, deform_envmap=deform_envmap)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # GT
        gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration > opt.start_aiaploss:
            # s_loss = torch.abs(torch.min(gaussians.get_scaling + d_scaling, dim=1).values).mean()
            # s_loss = torch.min(torch.abs(gaussians.get_scaling + d_scaling), dim=1).values.mean()
            # loss = loss + s_loss * opt.scale_loss
            loss = loss + 300000 * aiap_lossv2(gaussians.get_xyz.detach(), d_xyz)
        # lpips_los = calculate_lpips_loss(image, gt_image)
        # if iteration < 7000:
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lpips * lpips_los
        # else:
        #     if opt.dxyz != 0:    
        #         dxyz_los = reg_dxyz(d_xyz)
        #     else:
        #         dxyz_los = 0
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lpips * lpips_los + opt.dxyz * dxyz_los
        

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else \
                torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += REFL_MSK_LOSS_W * refl_msk_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()
            if iteration > 3000:
                # Log and save
                cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_dynamiccube_normal, (pipe, background),deform, env_deform)
                if (iteration in saving_iterations or iteration == TOT_ITER-1):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    deform.save_weights(args.model_path, iteration)
                    env_deform.save_weights(args.model_path, iteration)

                if iteration in testing_iterations:
                    if cur_psnr > best_psnr:
                        best_psnr = cur_psnr
                        best_iteration = iteration
                if iteration > 5000 and iteration %1000 ==0:
                    print("Current best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))
            # Densification
            if iteration < DENSIFY_UNTIL_ITER_NORAML:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= INIT_UNITIL_ITER:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= NORMAL_PROP_UNTIL_ITER:
                    opacity_reset_intval = 3000 # 2:1 (reset 1: reset 0)
                    densification_interval = DENSIFIDATION_INTERVAL_WHEN_PROP
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                
                if iteration > opt.densify_from_iter and iteration % densification_interval == 0 and iteration < DENSIFY_UNTIL_ITER:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 
                        opt.prune_opacity_threshold, 
                        scene.cameras_extent, size_threshold, 
                    )

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0(opt.resetv) # 不透明度小于0.01的不变，不透明度大于0.01的全部重置为0.01
                    gaussians.reset_refl(exclusive_msk=outside_msk) ### 把在球范围内外的高斯进行反射强度更新
                if  OPAC_LR0_INTERVAL > 0 and (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % OPAC_LR0_INTERVAL == 0: ## 200->50
                    gaussians.set_opacity_lr(opt.opacity_lr)
                
                # alldesify 多了这个
                if  (INIT_UNITIL_ITER < iteration <= NORMAL_PROP_UNTIL_ITER) and iteration % 1000 == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.dist_color(exclusive_msk=outside_msk) # dist_color 或者 reset_scale 会导致PSNR直线下降
                        # gaussians.reset_scale(exclusive_msk=outside_msk)
                        if OPAC_LR0_INTERVAL > 0 and iteration != NORMAL_PROP_UNTIL_ITER:
                            gaussians.set_opacity_lr(0.0)

            # Optimizer step
            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                env_deform.optimizer.step()
                env_deform.optimizer.zero_grad()
                # deform.update_learning_rate(iteration)
                if iteration > opt.warm_up:
                    deform.update_learning_rate(iteration)
                    env_deform.update_learning_rate(iteration)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        iteration += 1
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    deform, env_deform):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    cur_test = 0.0

    # Report test and samples of training set
    if iteration % 10_00 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        env_res = render_env_map(scene.gaussians)
        for env_name in env_res.keys():
            if tb_writer:
                tb_writer.add_image("#envmap/{}".format(env_name), env_res[env_name], global_step=iteration)
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    env_time_input = fid.unsqueeze(0).expand(viewpoint.HWK[0] * viewpoint.HWK[1], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    res = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, env_deform, env_time_input, more_debug_infos = True) # 这里可能有问题，在测试的时候会跳到后面去
                    image = torch.clamp(res["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        for maps_name in res.keys():
                            if 'map' in maps_name:
                                if 'normal' in maps_name:
                                     res[maps_name] = res[maps_name]*0.5+0.5
                                tb_writer.add_image(config['name'] + "_view_{}/{}".format(viewpoint.image_name, maps_name), res[maps_name], global_step=iteration)    
                        tb_writer.add_image(config['name'] + "_view_{}/2_render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == 10_000:
                            tb_writer.add_image(config['name'] + "_view_{}/1_ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if config['name']=='test': 
                        cur_test = psnr_test
                        # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        print("[ITER {}] The number of gaussians {}\n".format(iteration, len(xyz)), flush=True)
        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #tb_writer.add_scalar("refl_gauss_ratio", scene.gaussians.get_refl_strength_to_total.item(), iteration)
            
        torch.cuda.empty_cache()
    return cur_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000, 8000, 9000] + list(range(10000, 60001, 1000)))
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000, 60_000, 100_000, 150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 8_000, 9_000] + list(range(10000, 60001, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    print(args)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
