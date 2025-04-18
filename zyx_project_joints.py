import os
import smplx
import numpy as np
import open3d as o3d
import cv2
import json
import torch
import matplotlib.pyplot as plt

def get_smpl_joints(data):
    global_orient = torch.tensor(np.array(data['pred_smpl_params']['global_orient']), dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(np.array(data['pred_smpl_params']['body_pose']), dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(np.array(data['pred_smpl_params']['betas']), dtype=torch.float32).unsqueeze(0)

    pred_smpl_params = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
    }

    smpl_cfg = {
        'model_path': os.path.expanduser('~/.cache/4DHumans/data/smpl'),
        'gender': 'neutral',
    }
    smpl = smplx.SMPLLayer(**smpl_cfg)
    smpl_output = smpl(**{k: v for k, v in pred_smpl_params.items()}, pose2rot=False)
    return smpl_output.joints[0, :24]


def project_to_2D(points, intrinsic, extrinsic):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    cam_coords = extrinsic @ points_h.T
    proj = intrinsic @ cam_coords[:3, :]
    proj /= proj[2:3, :]
    return proj[:2, :].T


def backproject_keypoints(smpl_2d, point_cloud, img_size):
    pts = np.asarray(point_cloud.points)
    h, w = int(img_size[1]), int(img_size[0])
    print("raw point‑count:", pts.shape)
    print("expected points:", h*w)

    pts2d = pts.reshape((h, w, 3))

    back3d = []
    valid = []
    for x, y in smpl_2d:
        col = int(round(x))
        row = int(round(y))
        # Check bounds before indexing
        if 0 <= row < h and 0 <= col < w:
            p = pts2d[row, col]
            if not np.allclose(p, 0):
                back3d.append(p)
                valid.append(1)
            else:
                back3d.append([0, 0, 0])
                valid.append(0)
        else:
            # out-of-image coordinate
            back3d.append([0, 0, 0])
            valid.append(0)
    return np.array(back3d), np.array(valid)


if __name__ == '__main__':
    smpl_path = '/home/yuxuanzhang/Documents/Projects/4D-Humans/demo_out/20250418005607/10639122_0.json'
    pc_path = '/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250418005607/point_clouds/10639122.ply'

    # Load SMPL JSON
    data = json.load(open(smpl_path, 'r'))
    joints3d = get_smpl_joints(data)
    cam_t = np.array(data['pred_cam_t'])
    joints3d += cam_t
    focal = float(data['scaled_focal_length'])
    img_size = np.array([1920, 1080])

    # Intrinsic & Extrinsic
    center = img_size / 2.0
    intrinsic = np.array([[focal, 0, center[0]],
                          [0, focal, center[1]],
                          [0, 0, 1]])
    extrinsic = np.eye(4)

    # DEBUG
    joints_h  = np.hstack([joints3d, np.ones((24,1))])
    cam_coords = extrinsic @ joints_h.T
    print("Z range:", cam_coords[2].min(), "–", cam_coords[2].max())

    # Project to 2D
    joints2d = project_to_2D(joints3d, intrinsic, extrinsic)
    print(joints2d)

    # Load point cloud
    pc = o3d.io.read_point_cloud(pc_path)
    pts = np.asarray(pc.points) / 1000.0
    h, w = int(img_size[1]), int(img_size[0])
    pc.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))

    # Colorize
    img = cv2.imread('/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250418005607/color_images/10639122.png')
    colors = img.astype(np.float32) / 255.0
    pc.colors = o3d.utility.Vector3dVector(colors[..., ::-1].reshape(-1, 3))

    # Backproject
    back3d, valid = backproject_keypoints(joints2d, pc, img_size)

    # print("Valid keypoints:", valid)
    # plt.imshow(img[:, :, ::-1])
    # plt.scatter(joints2d[:, 0], joints2d[:, 1], c='r')
    # plt.title('Projected SMPL joints')
    # plt.show()

    # back3d : (24,3) array of joint positions
    # valid : (24,)  array of 0/1 flags

    # Separate valid vs invalid
    valid_pts   = back3d[valid == 1]
    invalid_pts = back3d[valid == 0]

    # Build two small point‐clouds for the joints
    pc_joints_valid = o3d.geometry.PointCloud()
    pc_joints_valid.points = o3d.utility.Vector3dVector(valid_pts)
    pc_joints_valid.paint_uniform_color([1.0, 0.0, 0.0])   # red

    pc_joints_invalid = o3d.geometry.PointCloud()
    pc_joints_invalid.points = o3d.utility.Vector3dVector(invalid_pts)
    pc_joints_invalid.paint_uniform_color([0.5, 0.5, 0.5]) # grey

    # Create a Visualizer to control point size
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PC + Joints', width=1280, height=720)
    vis.add_geometry(pc)

    # 2) For each valid joint, add a red sphere; for each invalid, a grey one
    sphere_radius = 0.02   # adjust to taste (in the same units as your point cloud!)
    for pt, v in zip(back3d, valid):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        mesh.translate(pt)   # move sphere to joint position
        if v:
            mesh.paint_uniform_color([1.0, 0.0, 0.0])  # red
        else:
            mesh.paint_uniform_color([0.5, 0.5, 0.5])  # grey
        mesh.compute_vertex_normals()  # helps with lighting
        vis.add_geometry(mesh)

    # 3) Leave render options at defaults (main cloud will stay normal‐sized)
    vis.run()
    vis.destroy_window()
