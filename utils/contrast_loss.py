import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)

        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = fg_img_feature.shape[0]
        for i in range(batch_size):
            curr_fg_img = fg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]
            
            fg_img_fg_pro_logits = curr_fg_img @ curr_fg_pro.t()  # [1, 1]
            fg_img_bg_pro_logits = curr_fg_img @ curr_bg_pro.t()  # [1, L]
            
            positive_sims = positive_sims + torch.exp(fg_img_fg_pro_logits / self.temperature).sum()
            
            negative_sims = negative_sims + \
                           torch.exp(fg_img_fg_pro_logits / self.temperature).sum() + \
                           torch.exp(fg_img_bg_pro_logits / self.temperature).sum()
        loss = -torch.log(positive_sims / negative_sims)

        return loss


class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)

        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)  # [N, D]
        
        batch_size = bg_img_feature.shape[0]
        for i in range(batch_size):
            curr_bg_img = bg_img_feature[i:i+1]  # [1, D]
            curr_fg_pro = fg_pro_feature[i:i+1]  # [1, D]
            curr_bg_pro = bg_pro_feature[i]  # [L, D]

            bg_img_bg_pro_logits = curr_bg_img @ curr_bg_pro.t()  # [1, L]
            bg_img_fg_pro_logits = curr_bg_img @ curr_fg_pro.t()  # [1, 1]

            positive_sims = positive_sims + torch.exp(bg_img_bg_pro_logits / self.temperature).mean()

            negative_sims = negative_sims + \
                           torch.exp(bg_img_bg_pro_logits / self.temperature).mean() + \
                           torch.exp(bg_img_fg_pro_logits / self.temperature).sum()

        loss = -torch.log(positive_sims / negative_sims)

        return loss
