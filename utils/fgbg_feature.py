import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
import os
import torchvision.utils as vutils
import cv2
import torchvision.transforms.functional as TF
from PIL import Image

class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()

        self.alpha = alpha
        self.mask_cam = mask_cam

        print(f"MaskAdapter_DynamicThreshold:")
        print(f"  alpha: {alpha}")
        print(f"  mask_cam: {mask_cam}")

    def forward(self, x):
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[0]), torch.zeros_like(x[0]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)

        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

class FeatureExtractor:
    def __init__(self, mask_adapter, clip_size=224):
        self.mask_adapter = mask_adapter
        self.clip_size = clip_size
        
    def prepare_cam_mask(self, cam, N):
        cam_224 = F.interpolate(cam, (self.clip_size, self.clip_size), 
                              mode="bilinear", align_corners=True)
        cam_224 = cam_224.reshape(N * cam.size(1), 1, self.clip_size, self.clip_size)

        cam_224_mask = self.mask_adapter(cam_224)
        
        return cam_224, cam_224_mask
        
    def prepare_image(self, img):
        return F.interpolate(img, (self.clip_size, self.clip_size), 
                           mode="bilinear", align_corners=True)
        
    @torch.cuda.amp.autocast()
    def extract_features(self, img_224, cam_224, cam_224_mask, label):
        batch_indices, class_indices = torch.where(label == 1)
        
        img_selected = img_224[batch_indices]  # [N, C, H, W]
        cam_selected = cam_224[batch_indices, class_indices]  # [N, H, W]
        mask_selected = cam_224_mask[batch_indices, class_indices]  # [N, H, W]
        
        cam_expanded = cam_selected.unsqueeze(1)  # [N, 1, H, W]
        mask_expanded = mask_selected.unsqueeze(1)  # [N, 1, H, W]
        
        fg_features = cam_expanded * img_selected  # [N, C, H, W]
        bg_features = (1 - cam_expanded) * img_selected  # [N, C, H, W]
        
        fg_masks = mask_expanded  # [N, 1, H, W]
        bg_masks = 1 - mask_expanded  # [N, 1, H, W]
        
        return fg_features, bg_features, fg_masks, bg_masks
        
    @torch.cuda.amp.autocast()
    def get_masked_features(self, fg_features, bg_features, fg_masks, bg_masks, clip_model):
        
        fg_min = fg_features.amin(dim=(2, 3), keepdim=True)
        fg_max = fg_features.amax(dim=(2, 3), keepdim=True)
        normalized_fg_features = (fg_features - fg_min) / (fg_max - fg_min + 1e-8)
        
        bg_min = bg_features.amin(dim=(2, 3), keepdim=True)
        bg_max = bg_features.amax(dim=(2, 3), keepdim=True)
        normalized_bg_features = (bg_features - bg_min) / (bg_max - bg_min + 1e-8)
        
        fg_img_features = clip_model.vision_model(normalized_fg_features*fg_masks)
        bg_img_features = clip_model.vision_model(normalized_bg_features*bg_masks)
            
        return fg_img_features, bg_img_features

        
    @torch.cuda.amp.autocast()
    def process_batch(self, inputs, cam, label, clip_model):
        if not torch.any(label == 1):
            return None
            
        cam_224 = F.interpolate(cam, (self.clip_size, self.clip_size), 
                               mode="bilinear", align_corners=True)
       
        cam_224_mask = self.mask_adapter(cam_224)
        
        fg_features, bg_features, fg_masks, bg_masks = self.extract_features(
            inputs, cam_224, cam_224_mask, label
        )
        
        fg_features, bg_features = self.get_masked_features(
            fg_features, bg_features, fg_masks, bg_masks, clip_model
        )
        
        return {
            'fg_features': fg_features,
            'bg_features': bg_features,
            'fg_masks': fg_masks,
            'bg_masks': bg_masks,
            'cam_224': cam_224,
            'cam_224_mask': cam_224_mask
        }

    def print_debug_info(self, batch_info):
        if batch_info is None:
            print("No foreground samples in batch")
            return
                
        print("\nFeature extraction debug info:")
        print(f"Number of foreground samples: {batch_info['fg_masks'].shape[0]}")
        print(f"Foreground features shape: {batch_info['fg_features'].shape}")
        print(f"Background features shape: {batch_info['bg_features'].shape}")
        print(f"CAM shape: {batch_info['cam_224'].shape}")
        print(f"Mask shape: {batch_info['cam_224_mask'].shape}")
        print(f"Foreground mask mean: {batch_info['fg_masks'].mean().item():.4f}")
        print(f"Background mask mean: {batch_info['bg_masks'].mean().item():.4f}")
