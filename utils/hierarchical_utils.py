import torch

def pair_features(fg_features, bg_features, text_features, labels):
    batch_indices, class_indices = torch.where(labels == 1)
    
    paired_fg_features = [] 
    paired_bg_features = [] 
    paired_fg_text = [] 
    paired_bg_text = [] 
 
    for i in range(len(batch_indices)):
        curr_class = class_indices[i]
        
        curr_fg = fg_features[i]  # [D]
        curr_bg = bg_features[i]  # [D]
        
        curr_fg_text = text_features[curr_class]  # [D]
        
        bg_text_indices = [j for j in range(text_features.shape[0]) if j != curr_class]
        curr_bg_text = text_features[bg_text_indices]  # [3, D]
        
        paired_fg_features.append(curr_fg)
        paired_bg_features.append(curr_bg)
        paired_fg_text.append(curr_fg_text)
        paired_bg_text.append(curr_bg_text)
    
    paired_fg_features = torch.stack(paired_fg_features)  # [N, D]
    paired_bg_features = torch.stack(paired_bg_features)  # [N, D]
    paired_fg_text = torch.stack(paired_fg_text)         # [N, D]
    paired_bg_text = torch.stack(paired_bg_text)         # [N, 3, D]
    
    return {
        'fg_features': paired_fg_features,  
        'bg_features': paired_bg_features,  
        'fg_text': paired_fg_text,         
        'bg_text': paired_bg_text       
    }


def merge_to_parent_predictions(predictions, k_list, method='max'):
    parent_preds = []
    start_idx = 0
    
    for k in k_list:
        if k > 1:
            class_preds = predictions[:, start_idx:start_idx + k]
            
            if method == 'max':
                class_probs = torch.softmax(class_preds, dim=1)
                parent_pred = (class_probs * class_preds).sum(dim=1)
            else:  # method == 'mean'
                parent_pred = torch.mean(class_preds, dim=1)
            
            parent_preds.append(parent_pred)
        else:
            parent_preds.append(predictions[:, start_idx])
        
        start_idx += k
    parent_preds = torch.stack(parent_preds, dim=1)
    
    return parent_preds


def merge_subclass_cams_to_parent(cams, k_list, method='max'):
    batch_size, _, H, W = cams.shape
    num_parent_classes = len(k_list)

    parent_cams = torch.zeros(batch_size, num_parent_classes, H, W, 
                            device=cams.device, dtype=cams.dtype)
    
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        if k > 1: 
            subclass_cams = cams[:, start_idx:start_idx + k, :, :]
            
            if method == 'max':
                B, k, H, W = subclass_cams.shape
                cams_flat = subclass_cams.view(B, k, H*W)  # [B, k, H*W]
                cams_probs = torch.softmax(cams_flat, dim=1)  # [B, k, H*W]
                parent_cam_flat = (cams_probs * cams_flat).sum(dim=1)  # [B, H*W]
                parent_cam = parent_cam_flat.view(B, H, W)  # [B, H, W]
            else:  # method == 'mean'
                parent_cam = torch.mean(subclass_cams, dim=1)    # [B, H, W]
            
            parent_cams[:, parent_idx, :, :] = parent_cam
        else:
            parent_cams[:, parent_idx, :, :] = cams[:, start_idx, :, :]
        
        start_idx += k
    
    return parent_cams


def expand_parent_to_subclass_labels(parent_labels, k_list):
    batch_size = parent_labels.size(0)
    total_subclasses = sum(k_list)
    
    subclass_labels = torch.zeros(batch_size, total_subclasses, 
                                device=parent_labels.device, 
                                dtype=parent_labels.dtype)
    start_idx = 0
    for parent_idx, k in enumerate(k_list):
        parent_label = parent_labels[:, parent_idx:parent_idx+1]  # [batch_size, 1]
        subclass_labels[:, start_idx:start_idx+k] = parent_label.repeat(1, k)
        start_idx += k
    
    return subclass_labels 