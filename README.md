# MRI Denoising with MONAI

Deep learning experiments for MRI denoising using MONAI (Medical Open Network for AI). This repository implements 3D UNet architectures for volumetric MRI denoising with comprehensive training, validation, and testing pipelines.

## Problem Statement

Magnetic Resonance Imaging (MRI) scans are subject to various noise sources including thermal noise, motion artifacts, and acquisition limitations. High-quality denoising is essential for:
- **Clinical Applications**: Improving diagnostic image quality
- **Research**: Enabling accurate quantitative analysis
- **Downstream Processing**: Enhancing segmentation and registration pipelines

This project implements state-of-the-art 3D UNet architectures using MONAI's medical imaging framework to address these challenges.

## Features

- **3D UNet Architecture**: Volumetric processing for full spatial context
- **MONAI Integration**: Leverages MONAI's medical imaging transforms and data loaders
- **Comprehensive Pipeline**: Training, validation, testing, and inference workflows
- **TensorBoard Logging**: Real-time training metrics and visualization
- **Checkpoint Management**: Model saving and resuming capabilities

## Project Structure

```
nn_MONAI/
├── driver.py            # Main entry point for training
├── model.py             # Model definition and training logic
├── Data.py              # Data loading and preprocessing
├── utils.py             # Utility functions
├── constants.py         # Configuration constants
└── requirements.txt     # Python dependencies
```

## Technical Details

### Architecture

- **Model**: 3D UNet from MONAI
- **Spatial Dimensions**: 3D (256×256×Z)
- **Channels**: (4, 8, 16, 32) with 8 residual units per level
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with StepLR scheduling (step_size=70, gamma=0.9)

### Data Processing

- **Input Format**: NIfTI files (.nii or .nii.gz)
- **Preprocessing**: Channel-first conversion, intensity scaling, random spatial cropping
- **Augmentation**: MONAI transforms for medical imaging

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
python driver.py \
    -lr 1e-3 \
    -epochs 100 \
    -dir_name /path/to/nii/files \
    -model_tag "UNet with Rand Crop" \
    -continue_training False
```

### Resume Training

```bash
python driver.py \
    -continue_training True \
    -callback_loc /path/to/saved/model
```

## Training Pipeline

1. **Data Loading**: Fetches NIfTI files and creates MONAI ImageDataset
2. **Data Splitting**: Automatic train/validation/test split
3. **Training Loop**: MSE loss optimization with validation monitoring
4. **Model Checkpointing**: Saves best model based on validation loss
5. **Evaluation**: Tests on held-out test set with quantitative metrics
6. **Inference**: Generates denoised outputs for visualization

## Results

The model is evaluated on:
- **Training/Validation Loss**: Tracked via TensorBoard
- **Test Set Metrics**: Quantitative evaluation on unseen data
- **Visual Outputs**: Denoised volumes saved for inspection

## License

This project uses MONAI, which is licensed under the Apache License 2.0.
