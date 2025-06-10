'''
python train_cls.py --config ./work_dirs/bcss/classification/config.yaml


'''
import argparse
import datetime
import os
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta
from skimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.trainutils import get_cls_dataset
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed,AverageMeter
from utils.evaluate import ConfusionMatrixAllClass
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils.hierarchical_utils import pair_features, merge_to_parent_predictions, merge_subclass_cams_to_parent, expand_parent_to_subclass_labels
from utils.validate import validate, generate_cam
from model.model import ClsNetwork
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
args = parser.parse_args()

def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and estimated time to completion"""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def train(cfg):
    
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto optimization
    
    num_workers = min(10, os.cpu_count())  # Optimize worker count based on CPU cores
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # Preload CLIP model to GPU and set to eval mode
    clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # ============ prepare data ============
    print("\nPreparing datasets...")
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    
    # Efficient data loading configuration
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            prefetch_factor=2,
                            persistent_workers=True)
    
    
    val_loader = DataLoader(val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    # ============ prepare model ============
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                    stride=cfg.model.backbone.stride,
                    cls_num_classes=cfg.dataset.cls_num_classes,
                    n_ratio=cfg.model.n_ratio,
                    pretrained=cfg.train.pretrained,
                    l_fea_path=cfg.model.label_feature_path)
    
    # Mixed precision training setup
    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    model.train()

    # Optimizer configuration
    optimizer = PolyWarmupAdamW(
        params=model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    # Loss functions and feature extractor setup
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha,)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    best_fuse234_dice = 0.0

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    
    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()
        
        with torch.cuda.amp.autocast():
            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list = model(inputs)

            # Merge subclass predictions to parent class predictions
            cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
            cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
            cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
            cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

            # Generate binary masks for feature extraction
            subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
            cls4_expand=expand_parent_to_subclass_labels(cls4_merge, k_list)
            cls4_bir=(cls4>cls4_expand).float()*subclass_labels

            # Extract foreground and background features
            batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, clip_model)
            fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']

            # Pair features with text embeddings
            set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
            fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                
            # Compute contrastive losses
            fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
            bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
            
            # Multi-scale classification losses
            loss1 = loss_function(cls1_merge, cls_labels)
            loss2 = loss_function(cls2_merge, cls_labels)
            loss3 = loss_function(cls3_merge, cls_labels)
            loss4 = loss_function(cls4_merge, cls_labels)
            cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
            loss = cls_loss + (fg_loss + bg_loss + 0.0005*torch.mean(cam4)) * cfg.train.l5

        # Gradient scaling for mixed precision
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing 5358
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            
            # Synchronize for accurate timing measurement
            torch.cuda.synchronize()
            
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
            )
        # Regular validation and model saving
        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss = validate(
                model=model,
                data_loader=val_loader,
                cfg=cfg,
                cls_loss_func=loss_function
            )   
            print("Validation results:")
            print(f"Val all acc4: {val_all_acc4:.6f}")
            print(f"Val avg acc4: {val_avg_acc4:.6f}")
            print(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
            
            if fuse234_score[:-1].mean() > best_fuse234_dice:
                best_fuse234_dice = fuse234_score[:-1].mean()
                save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
                
                # Save best model checkpoint
                torch.save(
                    {
                        "cfg": cfg,
                        "iter": n_iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    save_path,
                    _use_new_zipfile_serialization=True
                )
                print(f"\nSaved best model with mIOU: {best_fuse234_dice:.4f}")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    print(f'Total training time: {total_training_time}')

    
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)
 
    print("\nPreparing test dataset...")

    train_dataset, test_dataset = get_cls_dataset(cfg, split="test",enable_rotation=False,p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    

    test_loader = DataLoader(test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True)

    print("\n1. Testing on test dataset...")
    print("-" * 50)
    
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(
                model=model,
                data_loader=test_loader,
                cfg=cfg,
                cls_loss_func=loss_function)   

    
    print("Testing results:")
    print(f"Test all acc4: {test_all_acc4:.6f}")
    print(f"Test avg acc4: {test_avg_acc4:.6f}")
    print(f"Fuse234 score: {fuse234_score}, mIOU: {fuse234_score[:-1].mean():.4f}")
    
    print("\nPer-class IoU scores:")
    for i, score in enumerate(fuse234_score[:-1]):
        print(f"  Class {i}: {score:.6f}")
    if len(fuse234_score) > cfg.dataset.cls_num_classes:
        print(f"  Background: {fuse234_score[-1]:.6f}")

    print("\n2. Generating CAMs for complete training dataset...")
    print("-" * 50)
    
    train_cam_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.work_dir.pred_dir}")

    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        print(f"✓ Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        print("⚠ Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")
    
    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)
    
    print("\nFiles generated:")
    print(f"  • Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  • Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("="*80)
    
    
if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print('\nArgs: %s' % args)
    print('\nConfigs: %s' % cfg)

    set_seed(0)
    train(cfg=cfg)
