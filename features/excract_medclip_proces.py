import os
import torch
from PIL import Image
import pickle as pkl
from torchvision import transforms
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from tqdm import tqdm
import cv2 as cv
from utils.pyutils import set_seed
from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589, 0.22564577, 0.19820057]
    
    transform = A.Compose([
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        ToTensorV2(transpose_mask=True),
    ])
    return transform

def extract_features(image_dir):
    set_seed(42)
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    features_dict = {}
    for class_name in ['NEC', 'LYM', 'STR', 'TUM']:
        class_path = os.path.join(image_dir, class_name)
        features_dict[class_name] = []
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\nProcessing {class_name} images...")
        for img_name in tqdm(image_files, desc=f"{class_name}", ncols=100):
            img_path = os.path.join(class_path, img_name)
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

            transform = get_transform()
            img = transform(image=img)["image"]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            with torch.no_grad():
                outputs = model.vision_model(img.unsqueeze(0).to(device))
                features = outputs.cpu().detach().numpy()
            
            features_dict[class_name].append({
                'name': img_name,
                'features': features
            })
    
    save_dir = "./features/image_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "bcss_features_pro.pkl")
    
    with open(save_path, 'wb') as f:
        pkl.dump(features_dict, f)
    
    print(f"\nFeatures saved to {save_path}")

if __name__ == "__main__":
    image_dir = "data/BCSS-WSSS/proto"
    extract_features(image_dir)