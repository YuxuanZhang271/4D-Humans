import json
import numpy as np
import torch
import smplx
import open3d as o3d
from scipy.spatial.transform import Rotation as R

json_path = "demo_out/20250412070544/4419404_0.json"  # Path to your JSON file
with open(json_path, "r") as f:
    data = json.load(f)

smpl_param = data["pred_smpl_params"]
camera_t = data["pred_cam_t"]

global_orient_arr = np.array(smpl_param["global_orient"])  # shape: (1, 3, 3)
rot_matrix = global_orient_arr[0]  # Extract the 3x3 matrix
global_orient = R.from_matrix(rot_matrix).as_rotvec().reshape(1, 3)

body_pose_array = np.array(smpl_param["body_pose"])  # shape: (23, 3, 3)
body_pose_rotvec = np.array([R.from_matrix(mat).as_rotvec() for mat in body_pose_array])
body_pose_rotvec = body_pose_rotvec[:21, :]
body_pose_rotvec = body_pose_rotvec[np.newaxis, ...]

betas = np.array(smpl_param["betas"]).reshape(1, -1)   # shape: (1, 10)
transl = np.array(camera_t).reshape(1, 3)                # shape: (1, 3)

global_orient = torch.tensor(global_orient, dtype=torch.float32)
body_pose_rotvec = torch.tensor(body_pose_rotvec, dtype=torch.float32)
betas = torch.tensor(betas, dtype=torch.float32)
transl = torch.tensor(transl, dtype=torch.float32)

pose = {
    "global_orient": global_orient,  # (1, 3)
    "transl": transl,                # (1, 3)
    "body_pose": body_pose_rotvec,   # (1, 21, 3)
    "betas": betas                   # (1, 10)
}

model_path = "./"  # Update this to your SMPL-X model directory if needed
body_mesh_model = smplx.create(
    model_path,
    model_type='smplx',
    gender='neutral',
    ext='npz',
    num_pca_comps=12,
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_expression=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_transl=True,
    batch_size=1,
    num_betas=10,
    num_expression_coeffs=10
)

smplx_output = body_mesh_model(return_verts=True, **pose)
vertices = smplx_output.vertices[0].detach().cpu().numpy()  # shape: (N, 3)
faces = body_mesh_model.faces                                # shape: (M, 3)

# ---------------------------
# Correct the model orientation (flip upside down)
# ---------------------------
flip_R = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
vertices = (flip_R @ vertices.T).T  # Now vertices should be reoriented correctly

smpl_o3d_mesh = o3d.geometry.TriangleMesh()
smpl_o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
smpl_o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

smpl_o3d_mesh.paint_uniform_color([0.65098039,  0.74117647,  0.85882353])

smpl_o3d_mesh.compute_vertex_normals()  

o3d.visualization.draw_geometries(
    [smpl_o3d_mesh],
    window_name="SMPL-X Mesh with Normals",
    width=800,
    height=600
)
