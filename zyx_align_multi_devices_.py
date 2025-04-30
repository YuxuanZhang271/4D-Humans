import argparse
import cv2
import json
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import Camera
import os
import smplx
import torch
from typing import Optional, Sequence, Tuple


MODEL = 'models/smpl'

W, H = 1920, 1080
FX, FY = 1123.87, 1123.03 
CX, CY = 948.027, 539.649 

JOINTS_IDX = [0,1,2,3,4,5,6,7,8,9,10,11,12,15]
THRESH = 0.3


def config_from_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input folder')
    parser.add_argument('-t', '--timestamp', type=str, required=True, help='timestamp')
    args = parser.parse_args()
    return args


def project_joints(json_path): 
    # Load Data
    data = json.load(open(json_path, 'r'))
    pred_smpl_params = data['pred_smpl_params']
    global_orient = torch.tensor(pred_smpl_params['global_orient'], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.tensor(pred_smpl_params['body_pose'], dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(pred_smpl_params['betas'], dtype=torch.float32).unsqueeze(0)
    pred_cam_t_full = np.array(data['pred_cam_t_full'][0])
    scaled_focal_length = float(data['scaled_focal_length'])

    # Load Model
    smpl = smplx.SMPLLayer(
        model_path  = MODEL, 
        gender      = 'male'
    )
    pred_smpl_params = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
    }
    smpl_output = smpl(**{k: v for k, v in pred_smpl_params.items()}, pose2rot=False)
    joints = smpl_output.joints[0, JOINTS_IDX].detach().cpu().numpy()
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

    joints_h   = np.hstack([joints, np.ones((len(JOINTS_IDX), 1))])
    joints_cam = extrinsic @ joints_h.T
    joints_img = intrinsic @ joints_cam[:3, :]
    joints_img /= joints_img[2, :]

    return joints_img[:2, :].T, joints, verts, faces


def backproject_joints(depth, joints2d): 
    joints3d = []

    for u, v in joints2d:
        ui, vi = int(round(u)), int(round(v))

        if not (0 <= ui < W and 0 <= vi < H):
            joints3d.append((0, 0, 0))
            continue

        z = depth[vi, ui] / 1000.0
        x = (ui - CX) * z / FX
        y = (vi - CY) * z / FY
        joints3d.append((x, y, z))

    return np.stack(joints3d, axis=0)


def generate_pcd(depth, colors):
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    zs = depth / 1000.0
    xs = (us - CX) * zs / FX
    ys = (vs - CY) * zs / FY
    pts = np.stack((xs, ys, zs), axis=-1).reshape(-1,3)

    valid = (zs > 0).reshape(-1)
    pts = pts[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def refine_alignment_multi(j0_bp: np.ndarray,
                           j1_bp: np.ndarray,
                           T_ex: np.ndarray,
                           tol: float = 1e-2,
                           max_iters: int = 100
                           ) -> Tuple[np.ndarray, np.ndarray]:
    # Precompute extrinsics
    R_ex  = T_ex[:3, :3]
    t_ex  = T_ex[:3,  3]
    R_inv = R_ex.T

    # Build a fixed “valid” mask of joints that both cams actually saw
    j0_bp_cam0 = (R_inv @ (j0_bp - t_ex).T).T
    valid = (j0_bp_cam0[:,2] > 0) & (j1_bp[:,2] > 0)
    if valid.sum() < 3:
        raise ValueError(f"Need ≥3 valid joints, got {valid.sum()}")

    # Initialize both in cam1’s coords
    j0_ref = j0_bp.copy()
    j1_ref = j1_bp.copy()

    for it in range(1, max_iters+1):
        prev = np.vstack([j0_ref, j1_ref]).copy()

        # ———— bias in cam1 over the fixed valid set ————
        dz1    = j0_ref[valid,2] - j1_ref[valid,2]
        bias1  = np.mean(dz1)
        j1_ref[:,2] += bias1

        # ———— switch to cam0 ————
        j0_cam0 = (R_inv @ (j0_ref - t_ex).T).T
        j1_cam0 = (R_inv @ (j1_ref - t_ex).T).T

        # ———— bias in cam0 over the *same* valid indices ————
        dz0    = j1_cam0[valid,2] - j0_cam0[valid,2]
        bias0  = np.mean(dz0)
        j0_cam0[:,2] += bias0

        # ———— back to cam1 ————
        j0_ref = (R_ex @ j0_cam0.T).T + t_ex

        print(f"iter{it:2d}: bias0={bias0:.4f}, bias1={bias1:.4f}   ")

        # check convergence
        shift = np.abs(np.vstack([j0_ref, j1_ref]) - prev).max()
        if shift < tol:
            print(f"converged after {it:d} iterations (shift={shift:.4f})")
            break

    return j0_ref, j1_ref, valid


def rigid_transform_3D(
    A: np.ndarray,
    B: np.ndarray,
    valid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    A_valid = A[valid]
    B_valid = B[valid]
    if A_valid.shape[0] < 3:
        raise ValueError(
            f"At least 3 valid point correspondences required; "
            f"got {A_valid.shape[0]}"
        )

    centroid_A = A_valid.mean(axis=0)
    centroid_B = B_valid.mean(axis=0)
    AA = A_valid - centroid_A
    BB = B_valid - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A

    return R, t


def refine_alignment(R, t, joints3d, joints_bp, valid):
    # select only the original "valid" joints
    j3d_valid = joints3d[valid]
    jbp_valid = joints_bp[valid]

    # compute residuals and inlier mask
    residuals = np.linalg.norm((j3d_valid @ R.T + t) - jbp_valid, axis=1)
    inliers = residuals < THRESH

    # if too few inliers, bail out
    if inliers.sum() < 3:
        print("Too few inliers for refinement, using initial fit")
        return R, t

    # otherwise, build a mask over the *original* joints
    mask_refine = np.zeros_like(valid, dtype=bool)
    valid_indices = np.nonzero(valid)[0]      # e.g. array([0, 2, 3, 5, ...])
    # pick only those original indices that passed the inlier test
    mask_refine[ valid_indices[inliers] ] = True

    # now call rigid_transform_3D with three arguments
    R_refined, t_refined = rigid_transform_3D(
        joints3d,
        joints_bp,
        mask_refine
    )

    return R_refined, t_refined


def compute_mesh_scene_bias(pcd_scene: o3d.geometry.PointCloud,
                            mesh:    o3d.geometry.TriangleMesh,
                            n_samples: int = 20000
                           ) -> float:
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    d_scene_to_mesh = np.asarray(pcd_scene.compute_point_cloud_distance(mesh_pcd))
    d_mesh_to_scene = np.asarray(mesh_pcd.compute_point_cloud_distance(pcd_scene))

    return float(d_scene_to_mesh.mean()) + float(d_mesh_to_scene.mean())


def main(): 
    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    origin = '/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records'
    input = config_from_args().input
    record_time = os.path.basename(input)
    timestamp = config_from_args().timestamp


    # ----------------------------------------------------------------
    # Load Extrinsic
    # ----------------------------------------------------------------
    extrinsics_path = os.path.join(origin, record_time, 'stereo_extrinsics.json')
    with open(extrinsics_path, 'r') as f:
        ext = json.load(f)
    R_ex = np.array(ext['rotation_matrix'], dtype=np.float64)
    t_ex = np.array(ext['translation_vector'], dtype=np.float64)
    T_ex = np.eye(4)
    T_ex[:3, :3] = R_ex
    T_ex[:3,  3] = t_ex
    R_inv = R_ex.T


    # ----------------------------------------------------------------
    # Load Device 0
    # ----------------------------------------------------------------
    jpath_0 = f'{input}/device0/{timestamp}_0.json'
    joints2d_0, joints3d_0, verts_0, faces_0 = project_joints(jpath_0)

    mesh_0 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_0),
        o3d.utility.Vector3iVector(faces_0),
    )
    mesh_0.compute_vertex_normals()
    mesh_0.paint_uniform_color([0.7, 0.7, 0.9])

    depth_0_path = os.path.join(origin, record_time, 'device0/depth_images', f'{timestamp}.raw')
    depth_0 = np.fromfile(depth_0_path, dtype=np.float32).reshape((H, W))
    joints_bp_0 = backproject_joints(depth_0, joints2d_0)
    joints_bp_0 = (T_ex[:3, :3] @ joints_bp_0.T).T + T_ex[:3, 3]

    img_0_path = os.path.join(origin, record_time, 'device0/color_images', f'{timestamp}.png')
    img_0 = cv2.imread(img_0_path)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
    colors_0 = img_0.reshape(-1,3) / 255.0
    pcd_0 = generate_pcd(depth_0, colors_0)


    # ----------------------------------------------------------------
    # Load Device 1
    # ----------------------------------------------------------------
    jpath_1 = f'{input}/device1/{timestamp}_0.json'
    joints2d_1, joints3d_1, verts_1, faces_1 = project_joints(jpath_1)

    mesh_1 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_1),
        o3d.utility.Vector3iVector(faces_1),
    )
    mesh_1.compute_vertex_normals()
    mesh_1.paint_uniform_color([0.9, 0.7, 0.7])

    depth_1_path = os.path.join(origin, record_time, 'device1/depth_images', f'{timestamp}.raw')
    depth_1 = np.fromfile(depth_1_path, dtype=np.float32).reshape((H, W))
    joints_bp_1 = backproject_joints(depth_1, joints2d_1)

    img_1_path = os.path.join(origin, record_time, 'device1/color_images', f'{timestamp}.png')
    img_1 = cv2.imread(img_1_path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    colors_1 = img_1.reshape(-1,3) / 255.0
    pcd_1 = generate_pcd(depth_1, colors_1)


    # ----------------------------------------------------------------
    # Align Multi View
    # ----------------------------------------------------------------
    pcd_0.transform(T_ex)
    pcd_scene = pcd_0 + pcd_1

    joints3d_aligned_0, joints3d_aligned_1, valid = refine_alignment_multi(joints_bp_0, joints_bp_1, T_ex)

    joints3d_aligned_0_reset = (R_inv @ (joints3d_aligned_0 - t_ex).T).T
    R_refine_0, t_refine_0 = rigid_transform_3D(joints3d_0, joints3d_aligned_0_reset, valid)
    R_refine_0, t_refine_0 = refine_alignment(R_refine_0, t_refine_0, joints3d_0, joints3d_aligned_0, valid)
    T_0 = np.eye(4)
    T_0[:3, :3] = R_refine_0
    T_0[:3,  3] = t_refine_0
    mesh_0.transform(T_0)
    mesh_0.transform(T_ex)

    R_refine_1, t_refine_1 = rigid_transform_3D(joints3d_1, joints3d_aligned_1, valid)
    R_refine_1, t_refine_1 = refine_alignment(R_refine_1, t_refine_1, joints3d_1, joints3d_aligned_1, valid)
    T_1 = np.eye(4)
    T_1[:3, :3] = R_refine_1
    T_1[:3,  3] = t_refine_1
    mesh_1.transform(T_1)

    # joint_spheres_0 = []
    # for xyz in joints3d_aligned_0:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    #     sphere.translate(xyz)
    #     sphere.paint_uniform_color([0.7, 0.7, 0.9])
    #     joint_spheres_0.append(sphere)

    # joint_spheres_1 = []
    # for xyz in joints3d_aligned_1:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    #     sphere.translate(xyz)
    #     sphere.paint_uniform_color([0.9, 0.7, 0.7])
    #     joint_spheres_1.append(sphere)

    # ----------------------------------------------------------------
    # Align Multi View
    # ----------------------------------------------------------------
    # o3d.visualization.draw_geometries([pcd_scene, 
    #                                    mesh_0, mesh_1, 
    #                                    *joint_spheres_0, *joint_spheres_1
    #                                    ])
    if compute_mesh_scene_bias(pcd_scene, mesh_0, n_samples=20000) < compute_mesh_scene_bias(pcd_scene, mesh_1, n_samples=20000): 
        o3d.visualization.draw_geometries([pcd_scene, mesh_0])
    else: 
        o3d.visualization.draw_geometries([pcd_scene, mesh_1])


if __name__ == "__main__":
    main()