from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

import json

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

# Load your JSON data
with open('/home/yuxuanzhang/Documents/Projects/4D-Humans/demo_out/20250418005607/10639122_0.json') as f:
    data = json.load(f)
    pred_vertices = np.array(data['pred_vertices'])
    pred_cam_t = np.array(data['pred_cam_t'])
    focal_length_px = np.array(data['focal_length'])[0]

image = cv2.imread('/home/yuxuanzhang/Documents/Projects/pyorbbecsdk/records/20250418005607/color_images/10639122.png')
img_height, img_width, _ = image.shape

download_models(CACHE_DIR_4DHUMANS)
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
renderer = Renderer(model_cfg, faces=model.smpl.faces)
cam_view = renderer.render_rgba_multiple(
    vertices=[pred_vertices],  # your model vertices
    cam_t=[pred_cam_t],        # camera translation
    render_res=[img_width, img_height],
    focal_length=focal_length_px, 
    mesh_base_color=LIGHT_BLUE,
    scene_bg_color=(1, 1, 1),
)
input_img = image.astype(np.float32)[:,:,::-1]/255.0
input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

cv2.imshow('Rendered Image', input_img_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
