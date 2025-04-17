import json
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import smplx  # Ensure that you have installed the smplx library

# ------------------------------
# Step 1: Load Your JSON File
# ------------------------------
with open('demo_out/20250416012436/14705088_0.json', 'r') as f:
    data = json.load(f)

# Expected JSON structure:
# {
#   "pred_smpl_params": {
#       "global_orient": [[...3x3 matrix...]],
#       "body_pose": [[...], [...], ...],   # 23 total 3x3 matrices
#       "betas": [10 values]
#   },
#   "pred_cam_t": [tx, ty, tz]
# }
smpl_param = data['pred_smpl_params']
global_orient_matrix = np.array(smpl_param['global_orient'])
body_pose_matrices = np.array(smpl_param['body_pose'])  # Shape: (23, 3, 3)
betas = np.array(smpl_param['betas'])  # Shape: (10,)

# Extract camera translation
cam_t = np.array(data['pred_cam_t'])  # Should be a 3-element vector: [tx, ty, tz]
print("Camera translation (cam_t):", cam_t)

# Build an extrinsic matrix with cam_t (currently without any rotation)
extrinsic = np.eye(4)
extrinsic[:3, 3] = cam_t

# ------------------------------
# Step 2: Convert Rotations
# ------------------------------
# Convert the global orientation matrix to a 3D axis–angle vector.
global_orient = R.from_matrix(global_orient_matrix).as_rotvec()  # (3,)

# Convert each 3x3 body pose matrix into a 3D rotation vector.
body_pose = [R.from_matrix(mat).as_rotvec() for mat in body_pose_matrices]
body_pose = np.array(body_pose)  # Initially shape: (23, 3)

# Truncate the body pose to 21 joints (SMPLX expects (21, 3)).
if body_pose.shape[0] == 23:
    body_pose = body_pose[:21]

# ------------------------------
# Step 3: Load the SMPLX Model
# ------------------------------
model_path = './'
smplx_model = smplx.create(model_path=model_path,
                           model_type='smplx',
                           gender='male',
                           ext='npz',
                           use_pca=False)

# ------------------------------
# Step 4: Prepare Full SMPLX Parameters
# ------------------------------
# Set default zeros for parameters not available in SMPL.
left_hand_pose = np.zeros((15, 3))
right_hand_pose = np.zeros((15, 3))
jaw_pose = np.zeros((1, 3))
leye_pose = np.zeros((1, 3))
reye_pose = np.zeros((1, 3))
expression = np.zeros((10,))

# Convert parameters to torch tensors with a batch dimension.
global_orient = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
body_pose = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0)
betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
left_hand_pose = torch.tensor(left_hand_pose, dtype=torch.float32).unsqueeze(0)
right_hand_pose = torch.tensor(right_hand_pose, dtype=torch.float32).unsqueeze(0)
jaw_pose = torch.tensor(jaw_pose, dtype=torch.float32).unsqueeze(0)
leye_pose = torch.tensor(leye_pose, dtype=torch.float32).unsqueeze(0)
reye_pose = torch.tensor(reye_pose, dtype=torch.float32).unsqueeze(0)
expression = torch.tensor(expression, dtype=torch.float32).unsqueeze(0)

# ------------------------------
# Step 5: Inject Parameters and Get the Mesh
# ------------------------------
output = smplx_model(
    global_orient=global_orient,
    body_pose=body_pose,
    betas=betas,
    left_hand_pose=left_hand_pose,
    right_hand_pose=right_hand_pose,
    jaw_pose=jaw_pose,
    leye_pose=leye_pose,
    reye_pose=reye_pose,
    expression=expression,
    return_verts=True
)

# Get the mesh vertices and faces.
vertices = output.vertices.detach().cpu().numpy().squeeze(0)  # (N, 3)
faces = smplx_model.faces

# Optionally, if the SMPLX output provides joint locations, extract them.
# Many SMPLX implementations include an attribute "joints" in the output.
if hasattr(output, 'joints'):
    joints = output.joints.detach().cpu().numpy().squeeze(0)  # (num_joints, 3)
else:
    joints = None

# ------------------------------
# Step 6: Visualize the Mesh with Open3D and Display Joints
# ------------------------------
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

# Create a coordinate frame for reference.
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# Prepare a point cloud for the joints if available.
if joints is not None:
    joint_pcd = o3d.geometry.PointCloud()
    joint_pcd.points = o3d.utility.Vector3dVector(joints)
    # Set the joint point cloud color to red.
    joint_pcd.paint_uniform_color([1.0, 0.0, 0.0])
else:
    print("Warning: 'output' does not contain joints attribute.")

# Instead of directly using the extrinsic matrix from cam_t,
# we compute the mesh’s bounding box to set a “look-at” view.
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
print("Mesh center:", center)

# Set up the Visualizer.
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.add_geometry(coord_frame)
if joints is not None:
    vis.add_geometry(joint_pcd)

# Get current view control and camera parameters.
ctr = vis.get_view_control()
pinhole_params = ctr.convert_to_pinhole_camera_parameters()

# Option 1: Use a custom view that looks at the mesh center.
ctr.set_lookat(center)
ctr.set_front([0.0, 0.0, -1.0])   # Camera looks along negative Z-axis
ctr.set_up([0.0, -1.0, 0.0])       # Adjust up vector if needed
ctr.set_zoom(0.7)                 # Adjust zoom factor if needed

# Option 2: Alternatively, you could use the extrinsic matrix from cam_t.
# Uncomment the following two lines to use the extrinsic matrix instead:
# pinhole_params.extrinsic = extrinsic
# ctr.convert_from_pinhole_camera_parameters(pinhole_params)

print("Starting visualization. Close the window to exit.")
vis.run()
vis.destroy_window()
