# PBIP: Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation

**Official PyTorch implementation of the CVPR 2025 paper:**

> **Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation**  
> *CVPR 2025*

[📄 Paper](https://arxiv.org/abs/2503.12068)

---

## 🔥 Highlights

- 🏆 **CVPR 2025** acceptance
- 🎯 **Weakly Supervised Learning**: Achieves pixel-level segmentation using only image-level labels
- 🧬 **Histopathological Focus**: Specialized for medical image analysis
- 🚀 **Prototype-Based Design**: Novel prototype-based image prompting mechanism

## 🏗️ Model Architecture

<div align="center">
  <img src="Figure/model.png" alt="PBIP Model Architecture" width="800">
  <p><em>Overview of the PBIP architecture for weakly supervised histopathological image segmentation</em></p>
</div>

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)

### Environment Setup

#### Using requirements.txt
```bash
# Create virtual environment
conda create -n pbip python=3.8
conda activate pbip

# Install exact dependencies (recommended for reproducibility)
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the **BCSS (Breast Cancer Semantic Segmentation)** dataset with 5 tissue classes:

| Class | Description | Color |
|-------|-------------|-------|
| TUM | Tumor | 🔴 Red |
| STR | Stroma | 🟢 Green |
| LYM | Lymphocyte | 🔵 Blue |
| NEC | Necrosis | 🟣 Purple |
| BACK | Background | ⚪ White |

### Data Structure
```
data/
├── BCSS-WSSS/
│   ├── train/
│   │   └── *.png  # Training images with class labels in filename
│   ├── test/
│   │   ├── img/   # Test images
│   │   └── mask/  # Ground truth masks
│   └── valid/
│       ├── img/   # Validation images
│       └── mask/  # Ground truth masks
```

## 🚀 Quick Start
Downloading pre-trained SegFormer [here](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1).

Training Stage 1 & Generate CAMs
```bash
# Train the PBIP model
python train_stage_1.py --config ./work_dirs/bcss/classification/config.yaml --gpu 0
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```bibtex
@inproceedings{pbip2025,
  title={Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation},
  author={Qingchen Tang and Lei Fan and Maurice Pagnucco and Yang Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

⭐ **Star this repo if you find it helpful!** ⭐ 
