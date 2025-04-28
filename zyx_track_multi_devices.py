import argparse
import cv2
import glob
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full
import json
import numpy as np
import os
from pathlib import Path
import torch


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def config_from_args(): 
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('-i', '--img_folder', type=str, required=True, help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='output', help='Output folder to save rendered results')
    parser.add_argument('--detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')

    args = parser.parse_args()
    return args


def initialization(args): 
    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    return model, model_cfg, device, detector, renderer


def main(): 
    args = config_from_args()
    model, model_cfg, device, detector, renderer = initialization(args)

    root_dir = args.img_folder
    output_dir = args.out_folder
    os.makedirs(output_dir, exist_ok=True)
    output_sub_dir = os.path.join(args.out_folder, os.path.basename(root_dir))
    os.makedirs(output_sub_dir, exist_ok=False)

    device_dirs = [p for p in glob.glob(os.path.join(root_dir, 'device*')) if os.path.isdir(p)]
    for device_dir in device_dirs: 
        output_device_dir = os.path.join(output_sub_dir, os.path.basename(device_dir))
        os.makedirs(output_device_dir, exist_ok=False)

        image_dir = os.path.join(device_dir, 'color_images')
        image_paths = [p for p in glob.glob(os.path.join(image_dir, '*.png'))]
        for image_path in image_paths: 
            img_cv2 = cv2.imread(str(image_path))

            # Detect humans in image
            det_out = detector(img_cv2)

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)
                print("Model: ", out.keys())

                pred_cam = out['pred_cam']
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(image_path))
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()
                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                              out['pred_cam_t'][n].detach().cpu().numpy(),
                                              batch['img'][n],
                                              mesh_base_color=LIGHT_BLUE,
                                              scene_bg_color=(1, 1, 1),
                                            )
                    
                    final_img = np.concatenate([input_patch, regression_img], axis=1)
                    cv2.imwrite(os.path.join(output_device_dir, f'{img_fn}_{person_id}.png'), 255 * final_img[:, :, ::-1])

                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                    output_params = {}
                    for key, val in out.items():
                        if isinstance(val, dict):
                            nested = {}
                            for subk, subv in val.items():
                                nested[subk] = subv[n].detach().cpu().numpy().tolist()
                            output_params[key] = nested
                        else:
                            output_params[key] = val[n].detach().cpu().numpy().tolist()

                    output_params["scaled_focal_length"] = float(scaled_focal_length)
                    output_params["pred_cam_t_full"] = pred_cam_t_full.tolist()

                    json_path = os.path.join(output_device_dir, f'{img_fn}_{person_id}.json')
                    with open(json_path, 'w') as f:
                        json.dump(output_params, f, indent=4)
                    print(f"Saved JSON for {img_fn}{person_id}: {json_path}")


if __name__ == '__main__':
    main()
