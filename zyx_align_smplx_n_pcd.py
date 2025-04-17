import cv2
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import smplx
import torch


fx = 1123.87
fy = 1123.03
cx = 948.027
cy = 539.649
width = 1920 
height = 1080


def get_smpl_joints(data): 
    # Convert rotation matrices to axis-angle
    global_orient_mat = np.array(data['pred_smpl_params']['global_orient'])[0]  # (3, 3)
    global_orient = R.from_matrix(global_orient_mat).as_rotvec().reshape(1, 3)

    body_pose_mats = np.array(data['pred_smpl_params']['body_pose'])  # (23, 3, 3)
    body_pose = R.from_matrix(body_pose_mats).as_rotvec().reshape(1, 69)

    # Convert to tensors
    global_orient = torch.tensor(global_orient, dtype=torch.float32)
    body_pose = torch.tensor(body_pose, dtype=torch.float32)
    betas = torch.tensor(data['pred_smpl_params']['betas'], dtype=torch.float32).reshape(1, -1)
    transl = torch.tensor(data['pred_cam_t'], dtype=torch.float32).reshape(1, 3)
    transl[:, 2] /= 10.0

    model = smplx.create(
        model_path='models',
        model_type='smpl',
        gender='male',
        ext='pkl',
        batch_size=1
    )
    output = model(
        body_pose=body_pose,
        global_orient=global_orient,
        betas=betas,
        transl=transl
    )
    joints = output.joints.detach().cpu().numpy()[0]

    return joints[:24]


def project_2d_joints(smpl_joints):
    x = smpl_joints[:, 0]
    y = smpl_joints[:, 1]
    z = smpl_joints[:, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    return np.stack([u, v], axis=-1)  # shape: (N, 2)


def main(): 
    record_time = 20250418005607
    frame_time = 10642911 #10642911 10639122
    json_path = f'demo_out/{record_time}/{frame_time}_0.json'
    img_path = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{record_time}/color_images/{frame_time}.png'
    pcd_path = f'/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/{record_time}/point_clouds/{frame_time}.ply'

    with open(json_path, 'r') as f: 
        data = json.load(f)
    scale, tx, ty = np.array(data['pred_cam'])
    smpl_joints = get_smpl_joints(data)
    smpl_joints *= scale
    smpl_joints[:, 0] -= tx
    smpl_joints[:, 1] -= ty
    smpl_joints *= 1000.0
    print('smpl joints: \n', smpl_joints, '\n')

    joints_2d = project_2d_joints(smpl_joints)
    print('projected joints 2d: \n', joints_2d, '\n')
    
    img = cv2.imread(img_path)
    for joint in joints_2d:
        u, v = int(joint[0]), int(joint[1])
        cv2.circle(img, (u, v), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imshow('projected joints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pcd = o3d.io.read_point_cloud(pcd_path)

    smpl_joints_o3d = o3d.geometry.PointCloud()
    smpl_joints_o3d.points = o3d.utility.Vector3dVector(smpl_joints)
    red = np.array([[1.0, 0.0, 0.0]] * smpl_joints.shape[0])
    smpl_joints_o3d.colors = o3d.utility.Vector3dVector(red)

    pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0, 0, 0])
    smpl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100, origin=smpl_joints[0]  # or np.mean(smpl_joints, axis=0)
    )

    o3d.visualization.draw_geometries([pcd, smpl_joints_o3d, pcd_frame, smpl_frame])


if __name__ == "__main__":
    main()
