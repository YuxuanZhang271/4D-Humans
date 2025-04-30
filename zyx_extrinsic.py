import cv2
import glob
import json
import numpy as np
import os

# ┌──────────────────────────────────────────────────┐
# │ 1. SETTINGS                                     │
# └──────────────────────────────────────────────────┘
# Chessboard
board_size  = (10, 7)              # inner corners per (row, column)
square_size = 0.022                # in meters

# Termination criteria for cornerSubPix
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-6
)

# Paths
# — adjust this to the timestamped folder containing your two device subfolders
base_path = '/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250428120412'
imgsA = sorted(glob.glob(os.path.join(base_path, 'device0/color_images/*.png')))
imgsB = sorted(glob.glob(os.path.join(base_path, 'device1/color_images/*.png')))

print(f"[DEBUG] Found {len(imgsA)} images for device0, {len(imgsB)} images for device1")

# ┌──────────────────────────────────────────────────┐
# │ 2. PREPARE OBJECT POINTS                        │
# └──────────────────────────────────────────────────┘
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []   # 3D (same for both cameras)
img_pts_A  = []   # 2D in cam A
img_pts_B  = []   # 2D in cam B

# Flags for more robust corner detection
cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

# ┌──────────────────────────────────────────────────┐
# │ 3. DETECT & REFININE CHESSBOARD CORNERS         │
# └──────────────────────────────────────────────────┘
for fA, fB in zip(imgsA, imgsB):
    # Load & gray
    imgA, imgB = cv2.imread(fA), cv2.imread(fB)
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    okA, cornersA = cv2.findChessboardCornersSB(grayA, board_size,
                                                cv2.CALIB_CB_NORMALIZE_IMAGE | 
                                                cv2.CALIB_CB_EXHAUSTIVE)
    okB, cornersB = cv2.findChessboardCornersSB(grayB, board_size,
                                                cv2.CALIB_CB_NORMALIZE_IMAGE |
                                                cv2.CALIB_CB_EXHAUSTIVE)
    if not (okA and okB):
        continue

    # Refine
    cornersA = cv2.cornerSubPix(grayA, cornersA, (5,5), (-1,-1), criteria)
    cornersB = cv2.cornerSubPix(grayB, cornersB, (5,5), (-1,-1), criteria)

    obj_points.append(objp)
    img_pts_A.append(cornersA)
    img_pts_B.append(cornersB)

    vis = cv2.drawChessboardCorners(imgA.copy(), board_size, cornersA, okA)
    cv2.imshow('detected A', vis)
    cv2.waitKey(500)

print(f"[DEBUG] Valid stereo‐views detected: {len(obj_points)}")

if len(obj_points) == 0:
    raise RuntimeError("No valid chessboard detections — check board_size, image paths, or lighting.")

# ┌──────────────────────────────────────────────────┐
# │ 4. KNOWN INTRINSICS & DISTORTIONS               │
# └──────────────────────────────────────────────────┘
# From your Femto Bolt calibration:
fx, fy = 1123.87, 1123.03
cx, cy =  948.027, 539.649

# Intrinsic matrices
KA = np.array([[fx,   0, cx],
               [ 0,  fy, cy],
               [ 0,   0,  1]], dtype=np.float64)
KB = KA.copy()

# Distortion: [k1, k2, p1, p2, k3, k4, k5, k6]
DA = np.array([
    0.0733382, -0.101789,
   -0.000472246, -0.00022513,
    0.041689, 0, 0, 0
], dtype=np.float64)
DB = DA.copy()

# ┌──────────────────────────────────────────────────┐
# │ 5. STEREO CALIBRATION (EXTRINSICS ONLY)         │
# └──────────────────────────────────────────────────┘
flags = cv2.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_pts_A, img_pts_B,
    KA, DA, KB, DB,
    grayA.shape[::-1],
    criteria=criteria,
    flags=flags
)

print("Rotation (A → B):\n", R)
print("Translation (A → B) [m]:\n", T)

# ┌──────────────────────────────────────────────────┐
# │ 6. SAVE EXTRINSICS TO JSON                      │
# └──────────────────────────────────────────────────┘
extrinsics = {
    "rotation_matrix": R.tolist(),            # 3×3
    "translation_vector": T.flatten().tolist()  # 3×1 → [x, y, z]
}

out_path = os.path.join(base_path, "stereo_extrinsics.json")
with open(out_path, "w") as jf:
    json.dump(extrinsics, jf, indent=4)
print(f"Saved extrinsics to {out_path}")
