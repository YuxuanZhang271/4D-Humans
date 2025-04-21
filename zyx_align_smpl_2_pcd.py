import cv2
import json
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import Camera
import smplx
import torch


W = 1920 
H = 1080
fx, fy = 1123.87, 1123.03 
cx, cy = 948.027, 539.649 


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
    depth = np.fromfile(depth_path, dtype=np.uint16)
    depth = depth.reshape((H, W))  # HxW
    depth = depth.astype(np.float32)
    joints = []
    for u, v in joints2d: 
        us, vs = int(round(u)), int(round(v))
        zs = depth[vs][us] / 1000.0
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        joints.append(np.array([xs, ys, zs]))
    return np.stack(joints, axis=0)


def rigid_transform_3D(A: np.ndarray, B: np.ndarray):
    assert A.shape == B.shape
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # fix reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    
    return R, t


def main():
    record_time = 20250418005607
    timestamp = 10639122

    model_path = 'models/smpl'
    json_path = f'demo_out/{record_time}/{timestamp}_0.json'
    joints2d, joints, verts, faces = project_joints_2_img(model_path, json_path)

    # image = cv2.imread(f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{record_time}/color_images/{timestamp}.png')

    # img = image.copy()
    # for (u, v) in joints2d:
    #     cv2.circle(
    #         img,
    #         (int(round(u)), int(round(v))),  # pixel coords
    #         radius=2,                        # size of dot
    #         color=(0, 0, 255),               # BGR red
    #         thickness=-1                     # filled
    #     )

    # cv2.imshow("PointCloud + Joints", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    depth_path = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{record_time}/depth_images/{timestamp}.raw'
    joints3d = project_joints_2_pcd(depth_path, joints2d)
    # print(joints3d.shape)

    R, t = rigid_transform_3D(joints, joints3d)
    residuals = np.linalg.norm((joints @ R.T + t) - joints3d, axis=1)
    thresh = 0.5
    inliers = residuals < thresh
    if inliers.sum() >= 3:
        R_refined, t_refined = rigid_transform_3D(
            joints[inliers],
            joints3d[inliers]
        )
        R, t = R_refined, t_refined
    else:
        print("Too few inliers for refinement, using initial fit")

    backoff = -0.1
    v_forward = R[:, 2]  
    t_adjusted = t - v_forward * backoff
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t_adjusted
    
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

    point_cloud = o3d.io.read_point_cloud(f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{record_time}/point_clouds/{timestamp}.ply')
    pts = np.asarray(point_cloud.points)
    pts /= 1000.0
    point_cloud.points = o3d.utility.Vector3dVector(pts)

    o3d.visualization.draw_geometries([point_cloud, *joint_spheres, mesh])

if __name__ == '__main__': 
    main()
