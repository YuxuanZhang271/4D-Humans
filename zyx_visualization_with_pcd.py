import json
import numpy as np
import torch
import smplx
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------
# 1. Load the SMPL-X parameters from your JSON file.
# ---------------------------------------------------
json_path = "demo_out/20250412070544/4419404_0.json"  # Path to your JSON file
with open(json_path, "r") as f:
    data = json.load(f)

smpl_param = data["pred_smpl_params"]
camera_t = data["pred_cam_t"]

# ---------------------------
# 2. Process global orientation
# ---------------------------
global_orient_arr = np.array(smpl_param["global_orient"])  # shape: (1, 3, 3)
rot_matrix = global_orient_arr[0]  # Extract the 3x3 matrix
global_orient = R.from_matrix(rot_matrix).as_rotvec().reshape(1, 3)

# ---------------------------
# 3. Process body pose
# ---------------------------
body_pose_array = np.array(smpl_param["body_pose"])  # shape: (23, 3, 3)
body_pose_rotvec = np.array([R.from_matrix(mat).as_rotvec() for mat in body_pose_array])
body_pose_rotvec = body_pose_rotvec[:21, :]             # use first 21 joints
body_pose_rotvec = body_pose_rotvec[np.newaxis, ...]     # add batch dimension

# ---------------------------
# 4. Process betas and camera translation
# ---------------------------
betas = np.array(smpl_param["betas"]).reshape(1, -1)   # shape: (1, 10)
transl = np.array(camera_t).reshape(1, 3)                # shape: (1, 3)

# ---------------------------
# 5. Convert parameters to torch tensors
# ---------------------------
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

# ---------------------------
# 6. Initialize SMPL-X model
# ---------------------------
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
# 7. Correct the SMPL-X model orientation (flip upside down)
# ---------------------------
flip_R = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
# Apply to the mesh vertices:
vertices = (flip_R @ vertices.T).T

# Create the SMPL-X mesh as an Open3D TriangleMesh
smpl_o3d_mesh = o3d.geometry.TriangleMesh()
smpl_o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
smpl_o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
smpl_o3d_mesh.paint_uniform_color([0.65, 0.74, 0.86])
smpl_o3d_mesh.compute_vertex_normals()

# ---------------------------
# 8. Load the point cloud from a PLY file and prepare it
# ---------------------------
point_cloud_path = "/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250412070544/point_clouds/4419404.ply"
point_cloud = o3d.io.read_point_cloud(point_cloud_path)
# Apply a 6x scaling to the point cloud (if required by your dataset)
point_cloud.scale(6.0, center=point_cloud.get_center())

# Rotate the point cloud with the same flip_R so it aligns with the mesh
point_cloud.rotate(flip_R, center=point_cloud.get_center())

# ---------------------------
# 9. Inspect bounding boxes
# ---------------------------
mesh_bbox = smpl_o3d_mesh.get_axis_aligned_bounding_box()
pcd_bbox = point_cloud.get_axis_aligned_bounding_box()
print("Mesh bounding box min:", mesh_bbox.get_min_bound(), "max:", mesh_bbox.get_max_bound())
print("PointCloud bounding box min:", pcd_bbox.get_min_bound(), "max:", pcd_bbox.get_max_bound())

# ---------------------------
# 10. Scale and translate the mesh to align with the point cloud
# ---------------------------
# Compute extents (size along each axis)
mesh_extent = mesh_bbox.get_extent()
pcd_extent = pcd_bbox.get_extent()

# Compute a uniform scale factor by averaging the ratio (per axis)
scale_factors = np.array(pcd_extent) / np.array(mesh_extent)
uniform_scale_factor = scale_factors.mean()
print("Uniform scale factor:", uniform_scale_factor)

# Scale the mesh about its own center
mesh_center = smpl_o3d_mesh.get_center()
smpl_o3d_mesh.scale(uniform_scale_factor, center=mesh_center)

# After scaling, recompute the mesh center and translate so that it aligns with the point cloud center
new_mesh_center = smpl_o3d_mesh.get_center()
pcd_center = point_cloud.get_center()
translation = pcd_center - new_mesh_center
smpl_o3d_mesh.translate(translation)

# Optional: Print updated mesh center for confirmation
print("Updated mesh center:", smpl_o3d_mesh.get_center())
print("Point cloud center:", pcd_center)

# ---------------------------
# 11. Visualize both the mesh and the point cloud
# ---------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Aligned SMPL-X Mesh and Point Cloud", width=1200, height=800)

vis.add_geometry(smpl_o3d_mesh)
vis.add_geometry(point_cloud)

# Add a coordinate frame for reference
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
vis.add_geometry(coord_frame)

# Render options: adjust point size so that the point cloud doesn’t obscure the mesh
opt = vis.get_render_option()
opt.point_size = 2.0  
opt.mesh_show_back_face = True

vis.run()
vis.destroy_window()
