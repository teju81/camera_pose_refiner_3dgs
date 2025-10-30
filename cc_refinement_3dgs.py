import time
from PIL import Image
import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_renderer import render
from utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils import gui_utils
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.config_utils import load_config


from utils.camera_utils import Camera # Retain
from utils.pose_utils import update_pose # Retain function
from utils.slam_utils import get_loss_tracking, get_median_depth # Retain


class FrontEnd:
    def __init__(self, config):
        self.config = config # Retain
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # Retain
        self.pipeline_params = munchify(self.config["pipeline_params"]) # Retain
        self.gaussians = None

        self.device = "cuda:0" # Retain

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"] # Retain



    def tracking(self, cur_frame_idx, viewpoint):



        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                break


            # You have the updated Pose at this point - figure what you want to return from this point of the function

        return




    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        pass    

def main():

    # # Set up command line argument parser
    # parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument("--config", type=str)

    # args = parser.parse_args(sys.argv[1:])
    #config_file_path = args.config

    config_file_path = "/root/code/camera_pose_refiner_3dgs/config.yaml"

    with open(config_file_path, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    device = "cuda:0"

    # Intrinsics
    cx = 1256.19745
    cy = 843.384206
    fx = 2251.68856
    fy = 2326.39212
    H = 1440
    W = 2560

    theta_left  = np.arctan(cx / fx)
    theta_right = np.arctan((W - cx) / fx)
    theta_top   = np.arctan(cy / fy)
    theta_bot   = np.arctan((H - cy) / fy)

    fovx = np.degrees(theta_left + theta_right)
    fovy = np.degrees(theta_top  + theta_bot)

    # # Distortion Coefficients
    # distCoeffs = np.array([[-0.45140879,  0.22732696,  0., 0., 0. ]]) 


    # fx = config["Dataset"]["Calibration"]["fx"]
    # fy = config["Dataset"]["Calibration"]["fy"]
    # cx = config["Dataset"]["Calibration"]["cx"]
    # cy = config["Dataset"]["Calibration"]["cy"]
    # fovx = config["Dataset"]["Calibration"]["fovx"]
    # fovy = config["Dataset"]["Calibration"]["fovy"]
    # H = config["Dataset"]["Calibration"]["height"]
    # W = config["Dataset"]["Calibration"]["width"]


    # Need to define Viewpoint and pass it to this function
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=W,
        H=H,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device=device)

    # depth and pose can be some random initialization, gt_color must be initialized with infra camera RGB image
    gt_color = np.array(Image.open("/home/raviteja/code/datasets/artgarage/xgrids/camera_calibration/infra_cam_gt_rgb.png"))
    gt_depth = np.zeros((H, W), dtype=np.float32)
    gt_pose = np.eye(4, dtype=np.float32)

    viewpoint = Camera(
        0,
        gt_color,
        gt_depth,
        gt_pose,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        height,
        width,
        device=device,
    )
    viewpoint.compute_grad_mask(config)

    # Approx Extrinsics
    approx_ext = np.array([[ 0.34106683,  0.54607777,  0.76515773,  4.2593201 ],
     [-0.36391896,  0.82719075, -0.42820727,  0.29813336],
     [-0.86675423, -0.13242069,  0.48083896,  1.26206598],
     [ 0.,          0.,          0.,          1.        ]])
    R = approx_ext[:3,:3]
    T = approx_ext[:3, 3]

    # Need to provide approx infra camera pose here - check format
    viewpoint.update_RT(R, T)

    sh_degree = 3
    ply_file_path="/home/raviteja/code/datasets/artgarage/xgrids/camera_calibration/point_cloud.ply"

    # Load Gaussian Model
    use_spherical_harmonics = config["Training"]["spherical_harmonics"]
    model_params = munchify(config["model_params"])
    model_params.sh_degree = 3 if use_spherical_harmonics else 0
    front_end = FrontEnd(config)
    front_end.gaussians = GaussianModel(model_params.sh_degree, config=config) # Retain
    front_end.gaussians.load_plypath(ply_file_path)

    # Tracking
    render_pkg = front_end.tracking(cur_frame_idx, viewpoint)


        # curr_visibility = (render_pkg["n_touched"] > 0).long()

        # self.cleanup(cur_frame_idx)
        # toc.record()
        # torch.cuda.synchronize()




if __name__ == "__main__":
    main()
