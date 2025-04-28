import cv2
import json
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import Camera
import os
import smplx
import torch
from typing import Optional, Sequence, Tuple


model_path = 'models/smpl'

W, H = 1920, 1080
fx, fy = 1123.87, 1123.03 
cx, cy = 948.027, 539.649 

align_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,15]
thresh = 0.4


def project_joints_2_img(json_path): 
    data = json.load(open(json_path, 'r'))
    pred_smpl_params = data['pred_smpl_params']
    global_orient = torch.tensor(pred_smpl_params['global_orient'], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(pred_smpl_params['body_pose'], dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(pred_smpl_params['betas'], dtype=torch.float32).unsqueeze(0)
    pred_smpl_params = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
    }

    pred_cam_t_full = np.array(data['pred_cam_t_full'][0])
    scaled_focal_length = float(data['scaled_focal_length'])

    smpl = smplx.SMPLLayer(
        model_path  = model_path, 
        gender      = 'male'
    )
    smpl_output = smpl(**{k: v for k, v in pred_smpl_params.items()}, pose2rot=False)
    joints = smpl_output.joints[0, :24].detach().cpu().numpy()
    # joints[:, :2] += np.array([tx, ty])
    joints += pred_cam_t_full

    verts = smpl_output.vertices[0].detach().cpu().numpy()
    verts += pred_cam_t_full
    faces = smpl.faces.astype(np.int32)

    intrinsic = np.array([
        [scaled_focal_length,   0,                      W/2], 
        [0,                     scaled_focal_length,    H/2], 
        [0,                     0,                      1]
    ])
    extrinsic = np.eye(4)
    joints_h = np.hstack([joints, np.ones((joints.shape[0], 1))])
    joints_cam = extrinsic @ joints_h.T
    joints_img = intrinsic @ joints_cam[:3, :]
    joints_img /= joints_img[2, :]
    return joints_img[:2, :].T, joints, verts, faces


def project_joints_2_pcd(depth_path, joints2d):
    depth = (np.fromfile(depth_path, dtype=np.float32)
             .reshape((H, W)) 
             .astype(np.float32))
    
    joints3d = []
    for u, v in joints2d:
        ui, vi = int(round(u)), int(round(v))

        if not (0 <= ui < W and 0 <= vi < H):
            joints3d.append((0, 0, 0))
            continue

        z = depth[vi, ui] / 1000.0
        x = (ui - cx) * z / fx
        y = (vi - cy) * z / fy
        joints3d.append((x, y, z))

    return np.stack(joints3d, axis=0)


def rigid_transform_3D(
    A: np.ndarray,
    B: np.ndarray,
    keep_idx: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the best‐fit rotation R and translation t that maps A -> B (in a least‐squares sense),
    optionally only using a subset of point‐pairs.

    Args:
      A          (N×3 float): Source points (e.g. SMPL joints in camera frame).
      B          (N×3 float): Target points (e.g. back‐projected 3D joints).
      keep_idx   Optional list of indices into [0..N-1] of which correspondences to use.
                 If None, all pairs are considered, but any pairs where B[i]==(0,0,0) are dropped.

    Returns:
      R (3×3 float), t (3‐vector float) such that  R @ A[i] + t ≈ B[i].
    """
    # 1) Subset if requested
    if keep_idx is not None:
        A_sub = A[keep_idx]
        B_sub = B[keep_idx]
    else:
        A_sub = A.copy()
        B_sub = B.copy()

    # 2) Drop any invalid B (e.g. zeros from out‐of‐bounds)
    valid = ~np.all(B_sub == 0, axis=1)
    A_sub = A_sub[valid]
    B_sub = B_sub[valid]

    # 3) Need at least 3 correspondences
    if A_sub.shape[0] < 3:
        raise ValueError(f"Need ≥3 valid points, got {A_sub.shape[0]}")

    # 4) Centroids
    centroid_A = A_sub.mean(axis=0)
    centroid_B = B_sub.mean(axis=0)
    AA = A_sub - centroid_A
    BB = B_sub - centroid_B

    # 5) Correlation matrix and SVD
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 6) Fix reflection if needed
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 7) Translation
    t = centroid_B - R @ centroid_A

    return R, t


def load_pcd(root_dir, device, timestamp):
    # load color
    img = cv2.imread(f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/{device}/color_images/{timestamp}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = img.reshape(-1,3) / 255.0

    # load depth
    depth = np.fromfile(f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/{device}/depth_images/{timestamp}.raw', dtype=np.float32)
    depth = depth.reshape((H,W))
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    zs = depth / 1000.0
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    pts = np.stack((xs, ys, zs), axis=-1).reshape(-1,3)

    valid = (zs > 0).reshape(-1)
    pts = pts[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main(): 
    extrinsics_path = "stereo_extrinsics.json"
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    R_ex = np.array(ext['rotation_matrix'], dtype=np.float64)
    T_ex = np.array(ext['translation_vector'], dtype=np.float64)

    root_dir = 'output/20250428120412'
    timestamp = 131
    
    # device 0
    json_path_0 = f'{root_dir}/device0/{timestamp}_0.json'
    joints2d_0, joints_0, verts_0, faces_0 = project_joints_2_img(json_path_0)

    depth_path_0 = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/device0/depth_images/{timestamp}.raw'
    joints3d_0 = project_joints_2_pcd(depth_path_0, joints2d_0)

    R_0, t_0 = rigid_transform_3D(joints_0, joints3d_0, align_idx)
    residuals = np.linalg.norm((joints_0 @ R_0.T + t_0) - joints3d_0, axis=1)
    inliers = residuals < thresh
    if inliers.sum() >= 3:
        R_refined, t_refined = rigid_transform_3D(
            joints_0[inliers],
            joints3d_0[inliers]
        )
        R_0, t_0 = R_refined, t_refined
    else:
        print("Too few inliers for refinement, using initial fit")
    
    T_align = np.eye(4)
    T_align[:3, :3] = R_0
    T_align[:3,  3] = t_0

    T_extr = np.eye(4)
    T_extr[:3, :3] = R_ex
    T_extr[:3,  3] = T_ex

    T_0 = np.eye(4)
    T_0[:3, :3] = R_ex.dot(R_0)
    T_0[:3,  3] = R_ex.dot(t_0) + T_ex

    pcd_0 = load_pcd(root_dir, 'device0', timestamp)
    pcd_0.transform(T_extr)

    mesh_0 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_0),
        o3d.utility.Vector3iVector(faces_0),
    )
    mesh_0.compute_vertex_normals()
    mesh_0.transform(T_0)
    mesh_0.paint_uniform_color([0.5, 1.0, 0.5])

    # 1. Promote to homogeneous coordinates (N×4)
    j0_hom = np.hstack([
        joints3d_0,
        np.ones((joints3d_0.shape[0], 1))
    ])  # shape: (N,4)

    # 2. Transform into global frame
    j0_glob_hom = (T_extr @ j0_hom.T).T      # shape: (N,4)
    j0_glob     = j0_glob_hom[:, :3]         # drop the homogeneous 1s

    # 3. Now build your spheres at j0_glob
    joint_spheres_0 = []
    for xyz in j0_glob:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(xyz)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        joint_spheres_0.append(sphere)


    # device 1
    json_path_1 = f'{root_dir}/device1/{timestamp}_0.json'
    joints2d_1, joints_1, verts_1, faces_1 = project_joints_2_img(json_path_1)

    depth_path_1 = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/device1/depth_images/{timestamp}.raw'
    joints3d_1 = project_joints_2_pcd(depth_path_1, joints2d_1)

    R_1, t_1 = rigid_transform_3D(joints_1, joints3d_1, align_idx)
    residuals = np.linalg.norm((joints_1 @ R_1.T + t_1) - joints3d_1, axis=1)
    inliers = residuals < thresh
    if inliers.sum() >= 3:
        R_refined, t_refined = rigid_transform_3D(
            joints_1[inliers],
            joints3d_1[inliers]
        )
        R_1, t_1 = R_refined, t_refined
    else:
        print("Too few inliers for refinement, using initial fit")

    T_1 = np.eye(4)
    T_1[:3, :3] = R_1
    T_1[:3,  3] = t_1

    pcd_1 = load_pcd(root_dir, 'device1', timestamp)

    mesh_1 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_1),
        o3d.utility.Vector3iVector(faces_1),
    )
    mesh_1.compute_vertex_normals()
    mesh_1.transform(T_1)
    mesh_1.paint_uniform_color([1.0, 0.5, 0.5])

    joint_spheres_1 = []
    for xyz in joints3d_1:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(xyz)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        joint_spheres_1.append(sphere)
    

    # visualization
    o3d.visualization.draw_geometries([pcd_0, pcd_1, 
                                       *joint_spheres_0, *joint_spheres_1, 
                                       mesh_0, mesh_1
                                       ])

if __name__ == "__main__":
    main()
