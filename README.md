#  AI-Inpainting Region Detection with Attention U-Net

## Overview

In this project, we build a binary image segmentation system to detect AI-inpainted regions within images. Each pixel is classified as original (0) or manipulated (1). We design and train an Attention U-Net model using pairs of original images and their corresponding manipulation masks, incorporating data augmentation, custom loss, and CRF post-processing.

Our approach emphasizes model robustness, efficient training, and accurate evaluation with refined output masks.

---

##  Key Features

- **Attention U-Net** architecture for precise region segmentation
- **Train/Validation Split** with custom `InpaintDataset`
- **Test-Time Augmentation (TTA)** with flips and rotations
- **DenseCRF Post-Processing** to refine segmentation boundaries
- **Custom Hybrid Loss**: BCE + Dice Loss
- **GPU Support** via PyTorch
- **Automatic Model Saving** on best validation performance

---

##  Approach

### Data Preparation
- **Augmentation**:
  - Training: Resize, Horizontal/Vertical Flip, Rotation, ElasticTransform
  - Validation: Resize only
- **Normalization**: ImageNet-style stats (mean/std)
- **Image Format**: RGB (3-channel), 256x256
- **Mask Format**: Grayscale binary (0 or 1)

### Dataset Structure

```
train/
├── images/   # Manipulated images (inputs)
└── masks/    # Corresponding binary masks (labels)

test/
└── images/   # Images for prediction
```

### Model Architecture

```python
model = AttentionUNet(in_channels=3, out_channels=1)
```

- **Encoder**: Multi-stage CNN with skip connections
- **Attention Block**: Enhances skip connections between encoder and decoder
- **Final Layer**: 1-channel output with sigmoid activation

### Training Strategy

- Optimizer: `Adam` with LR = 1e-3
- Scheduler: `CosineAnnealingLR` over 50 epochs
- Loss: **BCE + Dice Loss**
- Batch Size: 16
- Early Stopping: Save best model by validation loss

### Inference Pipeline

- Apply `TTA` (flips, rotation)
- **Model output → Sigmoid → Threshold**
- **DenseCRF** post-processing on RGB image + probability map
- Convert binary mask to **Run-Length Encoding (RLE)** for submission

---


## File Descriptions

- `train_model.py`: Defines the model, dataset class, training loop, and loss function.
- `main.py`: Loads test images, applies TTA and CRF refinement, and generates `submission.csv`.
- `requirements.txt`: Python dependencies (see below).

---

## Requirements

```txt
numpy>=1.19
pandas>=1.1
opencv-python>=4.5
torch>=1.9
torchvision>=0.10
albumentations>=1.0
scikit-learn>=0.24
pydensecrf>=1.0
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### 1. Train the Model
```bash
python train_model.py
```

### 2. Generate Submission
```bash
python main.py
```

The script will produce a file named `submission.csv` containing the RLE-encoded masks.

---

## Sample Visualizations

- You can enhance this repo by visualizing:
  - Predicted vs. ground truth masks
  - Loss curves
  - DenseCRF effects
  - TTA results aggregation


