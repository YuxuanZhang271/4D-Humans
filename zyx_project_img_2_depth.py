import cv2
import numpy as np
import open3d as o3d

W = 1920 
H = 1080

fx, fy = 1123.87, 1123.03 
cx, cy = 948.027, 539.649 

# your extrinsic from device0 → device1
R = np.array([
    [ 0.71622868, -0.0529411 , -0.69585467], 
    [-0.05641483,  0.98946264, -0.1333456 ], 
    [ 0.69558166,  0.13476247,  0.70569486]
])
T = np.array([1.09121441, 0.23160971, 0.47697168])

def load_pcd(record_dir, device_id, timestamp):
    # load color
    img = cv2.imread(f'{record_dir}/device{device_id}/color_images/{timestamp}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = img.reshape(-1,3) / 255.0

    # load depth
    depth = np.fromfile(f'{record_dir}/device{device_id}/depth_images/{timestamp}.raw', dtype=np.float32)
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
    record_dir = '/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250425123027'
    stamp = '20'

    pcd0 = load_pcd(record_dir, 0, stamp)
    pcd1 = load_pcd(record_dir, 1, stamp)

    # build 4×4 extrinsic matrix
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3,:3] = R
    extrinsic[:3, 3] = T

    # transform device-0 → device-1 frame
    pcd0.transform(extrinsic)

    # color them differently if you like
    pcd0.paint_uniform_color([1,0,0])  # red
    pcd1.paint_uniform_color([0,1,0])  # green

    # visualize both together
    o3d.visualization.draw_geometries([pcd0, pcd1])

if __name__ == "__main__":
    main()
