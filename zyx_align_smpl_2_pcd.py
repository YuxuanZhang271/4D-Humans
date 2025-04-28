import cv2
import json
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import Camera
import smplx
import torch
import os
from typing import Optional, Sequence, Tuple


W = 1920 
H = 1080
fx, fy = 1123.87, 1123.03 
cx, cy = 948.027, 539.649 

align_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,15]


def project_joints_2_img(model_path, json_path): 
    data = json.load(open(json_path, 'r'))
    scale, tx, ty = data['pred_cam']
    pred_smpl_params = data['pred_smpl_params']
    global_orient = torch.tensor(pred_smpl_params['global_orient'], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(pred_smpl_params['body_pose'], dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(pred_smpl_params['betas'], dtype=torch.float32).unsqueeze(0)
    pred_smpl_params = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
    }

    pred_cam_t = np.array(data['pred_cam_t'])
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
    root_dir = 'output/20250425145208'
    timestamp = 144

    model_path = 'models/smpl'
    json_path = f'{root_dir}/device1/{timestamp}_0.json'
    joints2d, joints, verts, faces = project_joints_2_img(model_path, json_path)

    image = cv2.imread(f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/device1/color_images/{timestamp}.png')

    img = image.copy()
    for (u, v) in joints2d:
        cv2.circle(
            img,
            (int(round(u)), int(round(v))),  # pixel coords
            radius=2,                        # size of dot
            color=(0, 0, 255),               # BGR red
            thickness=-1                     # filled
        )

    cv2.imshow("PointCloud + Joints", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    depth_path = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{os.path.basename(root_dir)}/device1/depth_images/{timestamp}.raw'
    joints3d = project_joints_2_pcd(depth_path, joints2d)
    # print(joints3d.shape)

    R, t = rigid_transform_3D(joints, joints3d, align_idx)
    residuals = np.linalg.norm((joints @ R.T + t) - joints3d, axis=1)
    thresh = 0.4
    inliers = residuals < thresh
    if inliers.sum() >= 3:
        R_refined, t_refined = rigid_transform_3D(
            joints[inliers],
            joints3d[inliers]
        )
        R, t = R_refined, t_refined
    else:
        print("Too few inliers for refinement, using initial fit")

    # backoff = -0.1
    # v_forward = R[:, 2]  
    # t_adjusted = t - v_forward * backoff
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.transform(T)

    # visualization
    joint_spheres = []
    for xyz in joints3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(xyz)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        joint_spheres.append(sphere)

    pcd = load_pcd(root_dir, 'device1', timestamp)

    o3d.visualization.draw_geometries([pcd, *joint_spheres, mesh])

if __name__ == '__main__': 
    main()
