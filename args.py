#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import math
import os
from numbers import Integral


def _default_clidd_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLIDD-main")


GAUSSIAN_ABLATION_DEFAULTS = {
    "direct_gaussian_sampling": True,
    "pose_gaussian_gate": True,
    "pose_gaussian_min_update_count": 5,
    "pose_gaussian_min_observed_keyframes": 3,
    "densification_interval": 50,
    "densify_grad_threshold": 0.0002,
    "percent_dense": 0.01,
}
GAUSSIAN_ABLATION_OPTIONS = tuple(GAUSSIAN_ABLATION_DEFAULTS)


def _on_off(value):
    if value not in ("on", "off"):
        raise argparse.ArgumentTypeError("expected 'on' or 'off'")
    return value == "on"


def add_gaussian_ablation_arguments(parser):
    """Share Gaussian ablation switches between single-sequence and batch runs."""
    group = parser.add_argument_group("Gaussian ablation experiments")
    group.add_argument('--direct_gaussian_sampling', type=_on_off, metavar='{on,off}',
                       help="on: direct pixel sampling (default); off: sparse initialization followed by gradient-based Gaussian clone/split densification.")
    group.add_argument('--pose_gaussian_gate', type=_on_off, metavar='{on,off}',
                       help="Require both Gaussian maturity thresholds for pose optimization (default: on).")
    group.add_argument('--pose_gaussian_min_update_count', type=int,
                       help="Minimum visible Gaussian optimizer updates for pose optimization when the gate is on; nonnegative.")
    group.add_argument('--pose_gaussian_min_observed_keyframes', type=int,
                       help="Minimum distinct observing keyframes for pose optimization when the gate is on; nonnegative.")
    group.add_argument('--densification_interval', type=int,
                       help="Mapping optimizer steps between gradient densification passes when direct sampling is off; positive.")
    group.add_argument('--densify_grad_threshold', type=float,
                       help="Minimum accumulated mean screen-space gradient for Gaussian clone/split; finite and nonnegative.")
    group.add_argument('--percent_dense', type=float,
                       help="Gaussian size threshold relative to scene extent for choosing clone versus split; finite and positive.")
    group.set_defaults(**GAUSSIAN_ABLATION_DEFAULTS)


def validate_gaussian_ablation_arguments(args, parser):
    for name in ("pose_gaussian_min_update_count", "pose_gaussian_min_observed_keyframes"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if args.densification_interval <= 0:
        parser.error("--densification_interval must be positive")
    if not math.isfinite(args.densify_grad_threshold) or args.densify_grad_threshold < 0:
        parser.error("--densify_grad_threshold must be finite and nonnegative")
    if not math.isfinite(args.percent_dense) or args.percent_dense <= 0:
        parser.error("--percent_dense must be finite and positive")


def gaussian_ablation_config(args):
    """Return the actual typed configuration for logs and metric exports."""
    return {name: getattr(args, name, default) for name, default in GAUSSIAN_ABLATION_DEFAULTS.items()}


LOCAL_BA_VALUE_OPTIONS = (
    "local_ba_window_size",
    "local_ba_recent_keyframes",
    "local_ba_fixed_keyframes",
    "local_ba_max_points",
    "local_ba_min_shared_points",
    "local_ba_min_tracks",
    "local_ba_iterations",
    "local_ba_every",
    "local_ba_huber_delta",
    "local_ba_max_reprojection_error",
    "local_ba_min_parallax_deg",
    "local_ba_max_pose_change_ratio",
    "local_ba_max_rotation_deg",
)


def add_local_ba_arguments(parser):
    """Use identical covisibility BA options in single-sequence and batch runs."""
    group = parser.add_argument_group("Covisibility local bundle adjustment")
    group.add_argument('--local_ba_disable', action='store_true',
                       help="Disable joint local pose/landmark BA for an ablation run.")
    group.add_argument('--local_ba_window_size', type=int, default=10,
                       help="Maximum total local BA keyframes, including fixed anchors; at most 10.")
    group.add_argument('--local_ba_recent_keyframes', type=int, default=5,
                       help="Number of most recent valid training keyframes whose poses must be optimized; at least 5.")
    group.add_argument('--local_ba_fixed_keyframes', type=int, default=2,
                       help="Fixed boundary poses selected from older keyframes within the total window; at least 2 to constrain monocular scale.")
    group.add_argument('--local_ba_max_points', type=int, default=1500,
                       help="Maximum shared multi-view landmark tracks in one local BA solve.")
    group.add_argument('--local_ba_min_shared_points', type=int, default=30,
                       help="Minimum shared feature tracks for selecting a covisible keyframe.")
    group.add_argument('--local_ba_min_tracks', type=int, default=50,
                       help="Minimum valid landmark tracks required to run local BA.")
    group.add_argument('--local_ba_iterations', type=int, default=15,
                       help="Maximum nonlinear iterations for each local BA solve.")
    group.add_argument('--local_ba_every', type=int, default=5,
                       help="Run local BA every this many newly registered keyframes.")
    group.add_argument('--local_ba_huber_delta', type=float, default=3.0,
                       help="Huber transition in pixels for sparse reprojection residuals.")
    group.add_argument('--local_ba_max_reprojection_error', type=float, default=8.0,
                       help="Maximum reprojection error in pixels for a local BA observation.")
    group.add_argument('--local_ba_min_parallax_deg', type=float, default=1.0,
                       help="Minimum triangulation parallax in degrees for a shared landmark.")
    group.add_argument('--local_ba_max_pose_change_ratio', type=float, default=0.2,
                       help="Reject translations exceeding this fraction of local median landmark depth.")
    group.add_argument('--local_ba_max_rotation_deg', type=float, default=5.0,
                       help="Reject local BA pose updates exceeding this rotation in degrees.")


def validate_local_ba_window(window_size, recent_keyframes, fixed_keyframes):
    """Validate the same bounded window for CLI and direct selector calls."""
    for name, value in (
        ("local_ba_window_size", window_size),
        ("local_ba_recent_keyframes", recent_keyframes),
        ("local_ba_fixed_keyframes", fixed_keyframes),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"--{name} must be an integer")
    if recent_keyframes < 5:
        raise ValueError("--local_ba_recent_keyframes must be at least 5")
    if window_size > 10:
        raise ValueError("--local_ba_window_size must be at most 10, including fixed anchors")
    if fixed_keyframes < 2:
        raise ValueError("--local_ba_fixed_keyframes must be at least 2 to constrain monocular scale")
    if recent_keyframes + fixed_keyframes > window_size:
        raise ValueError(
            "--local_ba_recent_keyframes + --local_ba_fixed_keyframes "
            "must not exceed --local_ba_window_size"
        )


def validate_local_ba_arguments(args, parser):
    for name in LOCAL_BA_VALUE_OPTIONS:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    try:
        validate_local_ba_window(
            args.local_ba_window_size, args.local_ba_recent_keyframes,
            args.local_ba_fixed_keyframes,
        )
    except ValueError as exc:
        parser.error(str(exc))


def get_args():
    parser = argparse.ArgumentParser(description="Options for data loading and training")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to a dataset sequence or dataset root (KITTI: sequence/rgb/ and calib.txt; COLMAP: sparse/0/).")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--masks_dir', type=str, default="", 
                        help="If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png).")
    parser.add_argument('--num_loader_threads', type=int, default=4,
                        help="Number of workers to load and prepare input images")
    parser.add_argument('--downsampling', type=float, default=-1.0, help="Downsampling ratio for input images")
    parser.add_argument('--pyr_levels', type=int, default=2,
                        help="Number of pyramid levels. Each level l will downsample the image 2^l times in width and height")
    parser.add_argument('--min_displacement', type=float, default=0.02,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the dataset.")
    parser.add_argument('--dataset_type', choices=['auto', 'waymo', 'kitti', 'dl3dv', 'tum', 'nvs'], default='auto',
                        help="Dataset reader used by slam.py.")
    parser.add_argument('--dl3dv_quaternion_order', choices=['xyzw', 'wxyz'], default='xyzw',
                        help="Quaternion component order in DL3DV cameras.json (default: xyzw).")
    parser.add_argument('--kitti_camera', choices=['0', '1', '2', '3'], default='2',
                        help="Rectified KITTI camera: selects P0/P1/P2/P3 from calib.txt and image_0/1/2/3 when present. rgb/ defaults to the left color camera P2.")
    parser.add_argument('--camera_name', type=str, default='FRONT',
                        help="Waymo camera folder to read.")
    parser.add_argument('--waymo_image_model', choices=['auto', 'pinhole', 'opencv'], default='auto',
                        help="Waymo PNG/calibration contract: pinhole means already undistorted images with matching K; opencv means raw images to undistort with k1/k2/p1/p2/k3. auto rejects nonzero distortion because the PNG export cannot be inferred.")
    parser.add_argument('--max_frames', type=int, default=-1,
                        help="Optional cap on processed frames.")
    parser.add_argument('--export_da2_frames', nargs='+', default=[], metavar='FILENAME',
                        help="Also export official Depth Anything V2 relative predictions for these original image filenames before SLAM (e.g. 000094.png), into model_path/da2_depth.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device used by the SLAM pipeline.")
    parser.add_argument('--pose_format', choices=['auto', 'c2w', 'w2c'], default='auto',
                        help="Ground-truth pose convention (KITTI/Waymo/DL3DV auto assumes c2w).")
    parser.add_argument('--tum_max_dt', type=float, default=0.02,
                        help="Maximum TUM timestamp association error in seconds.")
    parser.add_argument('--tum_fx', type=float, default=-1.0)
    parser.add_argument('--tum_fy', type=float, default=-1.0)
    parser.add_argument('--tum_cx', type=float, default=-1.0)
    parser.add_argument('--tum_cy', type=float, default=-1.0)

    parser.add_argument('--sh_degree', type=int, default=3)

    ## COLMAP options
    parser.add_argument('--eval_poses', action='store_true',
                        help="Compare poses to COLMAP")
    parser.add_argument('--use_colmap_poses', action='store_true',
                        help="Load COLMAP data for pose and intrinsics initialization")
        
    ## Learning Rates
    parser.add_argument('--lr_poses', type=float, default=1.2e-4, help="Mapping pose learning rate")
    parser.add_argument('--lr_exposure', type=float, default=5e-4, 
                        help="Exposure compensation learning rate")
    parser.add_argument('--lr_depth_scale_offset', type=float, default=1e-4,
                        help="Learning rate for the monocular inverse-depth scale and offset during joint mapping")
    parser.add_argument('--position_lr_init', type=float, default=0.00005, 
                        help="Initial position learning rate")
    parser.add_argument('--position_lr_decay', type=float, default=1-2e-5, 
                        help="multiplicative decay factor for position learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.005, help="Feature learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="Opacity learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.01, help="Scaling learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.002, help="Rotation learning rate")
        
    ## Training schedule and losses
    parser.add_argument('--seed', type=int, default=0,
                        help="Global random seed for reproducible experiments.")
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="Weight for DSSIM loss")
    parser.add_argument('--num_iterations', type=int, default=30,
                        help="Legacy option; ordinary SLAM mapping samples 25 Gaussian-only steps followed by 25 joint pose/Gaussian steps")
    parser.add_argument('--slam_first_full_iterations', type=int, default=80,
                        help="Number of weighted random mapping optimization iterations during the post-bootstrap mapping pass")
    parser.add_argument('--slam_first_full_sampling_probs', type=float, nargs=8,
                        default=[0.22, 0.19, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03],
                        help="Sampling probabilities for the 8 bootstrap keyframes during the post-bootstrap mapping pass")
    parser.add_argument('--slam_final_extra_iterations', type=int, default=10,
                        help="Deprecated compatibility option, ignored; no optimization runs after online mapping ends")
    parser.add_argument('--depth_loss_weight_init', type=float, default=1e-2,
                        help="Initial per-keyframe weight of the monocular inverse-depth L1 loss")
    parser.add_argument('--depth_loss_weight_decay', type=float, default=0.9, 
                        help="Multiply the keyframe depth-loss weight by this factor after each mapping schedule step")
    parser.add_argument('--geometry_loss_weight', type=float, default=1.0,
                        help="Weight of the sparse reprojection and rendered inverse-depth consistency loss")
    parser.add_argument('--save_at_finetune_epoch', type=int, nargs='+', default=[], 
                        help="Enable finetuning after the initial on-the-fly reconstruction and save the scene at the end of the specified epochs when fine-tuning.")
    parser.add_argument('--use_last_frame_proba', type=float, default=0.2, 
                        help="Probability of using the last registered frame for each training iteration")

    ## Pose initialization options
    # Matching
    parser.add_argument('--num_kpts', type=int, default=8000,
                        help="Number of CLIDD sparse keypoints to extract from each image")
    parser.add_argument('--match_max_error', type=float, default=2e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--fundmat_samples', type=int, default=2000,
                        help="Maximum number of set of matches used to estimate the fundamental matrix for outlier removal")
    parser.add_argument('--min_num_inliers', type=int, default=100,
                        help="The keyframe will be added only if the number of inliers is at least this value")
    # Initial mini bundle adjustment
    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=200)
    parser.add_argument('--enable_reboot', action='store_true')
    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true', 
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    # Incremental pose optimization
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=6,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=20,
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--pnpransac_samples', type=int, default=2000,
                        help="Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=20)

    add_local_ba_arguments(parser)

    parser.add_argument('--clidd_root', type=str, default=_default_clidd_root(),
                        help="Path to CLIDD-main. Defaults to the CLIDD-main folder under this TEST2-SLAM checkout.")
    parser.add_argument('--clidd_model', type=str, default='S64',
                        choices=['A48', 'N64', 'T64', 'S64', 'M64', 'L64', 'G128', 'E128', 'U128'],
                        help="CLIDD model variant used for sparse and dense local features.")
    parser.add_argument('--clidd_radius', type=int, default=2,
                        help="Non-maximum suppression radius passed to CLIDD.")
    parser.add_argument('--clidd_score', type=float, default=-5.0,
                        help="CLIDD detection score threshold.")
    parser.add_argument('--clidd_match_beta', type=float, default=20.0,
                        help="Soft mutual matching beta used for CLIDD descriptors.")
    parser.add_argument('--clidd_dense_stride', type=int, default=8,
                        help="Pixel stride for CLIDD dense descriptor maps used by guided MVS.")
    add_gaussian_ablation_arguments(parser)

    ## Gaussian initialization options
    parser.add_argument('--init_sampling_method', choices=['scharr'], default='scharr',
                        help="Gaussian initialization image response (Scharr only; used by default).")
    parser.add_argument('--init_proba_scaler', type=float, default=1,
                        help="Scale the image-response probability of using a pixel to make a new Gaussian primitive. Set to 0 to only use triangulated points.")
    parser.add_argument('--slam_neighbor_penalty_weight', type=float, default=0.2,
                        help="Weight of the valid 8-neighbor rendered penalty in direct Gaussian sampling.")
    parser.add_argument('--mono_depth_backend', choices=['da2', 'da3mono'], default='da2',
                        help="Monocular depth backend. Defaults to da2. Passing --da3_model_id automatically enables da3mono.")
    parser.add_argument('--da3_model_id', type=str, default=None,
                        help="Depth Anything 3 model id or local model path. If set, TEST2-SLAM uses the da3mono backend.")

    # Anchor management
    parser.add_argument('--anchor_overlap', type=float, default=0.3,
                        help="Size of the overlapping regions when blending between anchors")

    ## Keyframe management
    parser.add_argument('--max_active_keyframes', type=int, default=200,
                        help="Maximum number of keyframes to keep in GPU memory. Will start offloading keyframes to CPU if this number is exceeded.")
    parser.add_argument('--keyframe_min_interval', type=int, default=2,
                        help="Deprecated compatibility option; keyframe selection no longer uses a minimum frame-id gap.")
    parser.add_argument('--slam_latest_keyframe_sample_proba', type=float, default=0.25,
                        help="Probability of sampling the latest keyframe during each SLAM mapping optimization step")
    parser.add_argument('--slam_window_sampling_seed', type=int, default=None,
                        help="Optional independent seed for random SLAM mapping keyframe sampling. Defaults to --seed.")

    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1,
                        help="Holdout for render metrics. For example, --test_hold 8 selects frame ids 8, 16, 24, ... for PSNR/SSIM/LPIPS and excludes selected keyframes from Gaussian optimization. ATE/RMSE ATE is computed on keyframes only. If set to -1, no render holdout frames are selected.")
    parser.add_argument('--eval_lpips_max_size', type=int, default=512,
                        help="Maximum long image side for LPIPS evaluation. Set <=0 to evaluate LPIPS at full resolution.")
    parser.add_argument('--eval_lpips_device', choices=['cpu', 'cuda'], default='cpu',
                        help="Device used for LPIPS evaluation. CPU avoids GPU OOM during high-resolution saves.")
    parser.add_argument('--test_frequency', type=int, default=-1, 
                        help="Test and get metrics every test_frequency keyframes")
    parser.add_argument('--display_runtimes', action='store_true', 
                        help="Display runtimes for each step in the tqdm bar")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Directory to store the renders from test view and checkpoints after training. If not set, will be set to results/xxxxxx.")
    parser.add_argument('--save_every', default=-1, type=int, 
                        help="Frequency of exporting renders w.r.t input frames.")

    ## Viewer
    parser.add_argument('--viewer_mode', choices=['local', 'server', 'web', 'none'], default='none',
                        help="Live HTTP viewer: web/server print the URL; local also opens a browser; none disables it.")
    parser.add_argument('--ip', type=str, default="127.0.0.1",
                        help="Viewer bind address. Use SSH forwarding for a remote server, or 0.0.0.0 for LAN access.")
    parser.add_argument('--port', type=int, default=6009,
                        help="HTTP port for the live viewer.")
    parser.add_argument('--viewer_interval', type=float, default=0.5,
                        help="Minimum seconds between live snapshots, captured at frame boundaries.")
    parser.add_argument('--viewer_max_points', type=int, default=20000,
                        help="Maximum displayed Gaussian centers; does not change the SLAM map.")
    parser.add_argument('--viewer_image_size', type=int, default=640,
                        help="Maximum long side of viewer input and Gaussian render images.")
    parser.add_argument('--viewer_keep_alive', action='store_true',
                        help="After saving results, keep the final web view available until Ctrl+C.")

    args = parser.parse_args()
    validate_local_ba_arguments(args, parser)
    validate_gaussian_ablation_arguments(args, parser)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.viewer_interval) or args.viewer_interval <= 0:
        parser.error("--viewer_interval must be finite and positive")
    if args.viewer_max_points <= 0:
        parser.error("--viewer_max_points must be positive")
    if args.viewer_image_size < 64:
        parser.error("--viewer_image_size must be at least 64")

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
