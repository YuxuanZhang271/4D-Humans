import os
import smplx
import argparse
import numpy as np
import open3d as o3d

import cv2
import json
import torch
from glob import glob
from natsort import natsorted

def get_smpl_joints(data):
        
    global_orient   = torch.tensor(np.array(data['pred_smpl_params']['global_orient']), dtype=torch.float32)    # [batch, 1,  3, 3]
    body_pose       = torch.tensor(np.array(data['pred_smpl_params']['body_pose']), dtype=torch.float32)        # [batch, 23, 3, 3]
    betas           = torch.tensor(np.array(data['pred_smpl_params']['betas']), dtype=torch.float32)            # [1,  10]
    
    pred_smpl_params = {}
    pred_smpl_params['global_orient']   = global_orient
    pred_smpl_params['body_pose']       = body_pose
    pred_smpl_params['betas']           = betas
    
    smpl_cfg = {'model_path': '/home/haziq/.cache/4DHumans/data/smpl', 
                'gender': 'neutral',
               }
            
    # forward pass
    # smpl still gives more joints than what we need so we need take the first 24 https://github.com/vchoutas/smplx/blob/main/smplx/vertex_joint_selector.py
    smpl        = smplx.SMPLLayer(**smpl_cfg)
    smpl_output = smpl(**{k:v for k,v in pred_smpl_params.items()}, pose2rot=False)
    xyz_joints  = smpl_output.joints#[0,:24]
                               
    return xyz_joints

def project_to_2D(points, intrinsic, extrinsic):
    """
    Projects 3D points to 2D using camera intrinsic and extrinsic matrices.

    Args:
        points (np.array): A Nx3 array of 3D points.
        intrinsic (np.array): A 3x3 intrinsic camera matrix.
        extrinsic (np.array): A 4x4 extrinsic camera matrix (rotation and translation).

    Returns:
        np.array: A Nx2 array of 2D points projected onto the image plane.
    """

    # Convert 3D points to homogeneous coordinates (Nx4 array)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply extrinsic transformation to get the points in the camera coordinate system
    camera_coords = extrinsic @ points_homogeneous.T

    # Apply intrinsic transformation to project to the image plane
    # The intrinsic matrix maps from 3D to 2D camera coordinates
    projected_points = intrinsic @ camera_coords[:3, :]  # We only need the first 3 rows (x, y, z)

    # Convert from homogeneous to 2D coordinates by dividing by the depth (z)
    projected_points /= projected_points[2, :]

    # Extract x and y coordinates (2D points)
    projected_2d = projected_points[:2, :].T

    return projected_2d

# given the coordinates of the keypoints in 2D, we get its 3D coordinates by referencing [row,col] from the point cloud
def backproject_keypoints(smpl_2d_keypoints, point_cloud, smpl_img_size):
        
    # Convert point cloud to numpy array and reshape to image dimensions
    points = np.asarray(point_cloud.points)
    points = np.reshape(points, (int(smpl_img_size[1]), int(smpl_img_size[0]), -1))

    # Initialize arrays for the backprojected 3D keypoints and their validity
    backprojected_keypoints = []
    valid_keypoints = []

    for idx, keypoint in enumerate(smpl_2d_keypoints):
        row, col = int(keypoint[1]), int(keypoint[0])  # row corresponds to y, col to x
        point = points[row, col]  # Get the 3D point at the keypoint position

        # Check if the point has valid depth data
        if not np.all(point == 0):  
            backprojected_keypoints.append(point)
            valid_keypoints.append(1)  # Mark as valid
        else:
            backprojected_keypoints.append([0, 0, 0])  # Add a placeholder for missing points
            valid_keypoints.append(0)  # Mark as invalid

    return np.array(backprojected_keypoints), np.array(valid_keypoints)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    
    # assert check
    smpl_filenames  = natsorted(glob(os.path.join(args.folder_name,"4D-Humans","json","*")))
    cloud_filenames = natsorted(glob(os.path.join(args.folder_name,"point_cloud","*")))
    assert len(smpl_filenames) == len(cloud_filenames)
    
    """
    load data
    """
        
    range_file_idxs = range(len(smpl_filenames))    
    for file_idx in range_file_idxs:
        
        print(smpl_filenames[file_idx])
        print(cloud_filenames[file_idx])
        print()
        
        # load smpl data
        smpl_data                   = json.load(open(smpl_filenames[file_idx], "r"))
        smpl_3d_joints              = get_smpl_joints(smpl_data)                    # [1, 45, 3]
        smpl_3d_joints              = smpl_3d_joints[0,:24]                         # [24, 3]
        smpl_pred_cam_t_full        = np.array(smpl_data["pred_cam_t_full"])[0]     # [3]
        smpl_scaled_focal_length    = np.array(smpl_data["scaled_focal_length"])    # scalar
        smpl_img_size               = np.array(smpl_data["img_size"])[0]            # [2]
        
        # load image
        image                       = cv2.imread(smpl_data["img_path"].replace("image","rgb"))             # [720, 1280, 3]
        
        # joints absorb the translation
        smpl_3d_joints = smpl_3d_joints + smpl_pred_cam_t_full    #    [24, 3]
        
        # extrinsic - assume no translation nor rotation
        smpl_extrinsic   = np.eye(4)
        
        # intrinsic
        smpl_camera_center = np.array([smpl_img_size[0]/2., smpl_img_size[1]/2.])    # image center
        smpl_intrinsic = np.array([[smpl_scaled_focal_length, 0,                         smpl_camera_center[0]],
                                   [0,                        smpl_scaled_focal_length,  smpl_camera_center[1]],
                                   [0,                        0,                                            1]])
            
        # project to 2D
        smpl_2d_keypoints = project_to_2D(smpl_3d_joints, smpl_intrinsic, smpl_extrinsic)  # [24, 2]
        
        """
        get point cloud
        """
        
        # load point cloud
        point_cloud = o3d.io.read_point_cloud(cloud_filenames[file_idx])
        points = np.asarray(point_cloud.points)
        points = points / 1000.0  # Convert from mm to m ?
        points = np.reshape(points,[int(smpl_img_size[1]),int(smpl_img_size[0]),-1])
        point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        
        # add color to point cloud for visualization
        image  = cv2.imread(smpl_data["img_path"].replace("image","rgb"))
        colors = image.astype(np.float32) / 255.0
        colors = colors[..., ::-1]
        colors = np.reshape(colors, [-1, 3])
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # backproject
        backprojected_keypoints, valid_backprojected_keypoint_indices = backproject_keypoints(smpl_2d_keypoints, point_cloud, smpl_img_size)