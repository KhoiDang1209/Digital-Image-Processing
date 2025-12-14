# Sign Language Detection Module

A deep learning module for binary classification of sign language gestures using hand landmark detection and LSTM/Transformer models.

## Overview

This module detects whether a user is performing sign language gestures (YES) or not (NO) using MediaPipe hand landmark extraction and sequence classification with LSTM or Transformer architectures.

## Features

- Binary sign language classification (YES/NO)
- MediaPipe-based hand landmark extraction (up to 2 hands, 126 dimensions)
- LSTM with attention mechanism
- Transformer encoder architecture
- Configurable training pipeline with early stopping
- False negative penalty weighting for better recall
- Support for both video and frame-based input
- Cached dataset loading for faster training

## Project Structure

```
sign_language_detection_module/
├── main.py                      # CLI entry point
├── train.py                     # Training module with Trainer class
├── data.py                      # Dataset classes
├── transform.py                 # Video/Frame to numpy converters
├── preprocess.py                # SL and Jester dataset preprocessors
├── config/
│   ├── config.py               # Configuration classes
│   └── config.yaml             # Training configuration
├── data/
│   ├── SL_YES_npy/            # Sign language positive samples
│   ├── jester_NO_npy/         # Jester negative samples
│   └── splits/                # Train/val/test CSV splits
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs
├── results/                   # Training curves and results
└── README.md

```

## Usage

### Training

Train the model with default configuration:

```bash
python main.py train
```

Train with custom configuration:

```bash
python main.py train --config config/config.yaml --fn-penalty 2.0
```

Options:
- `--config`: Path to config YAML file (default: `config/config.yaml`)
- `--use-cache` / `--no-cache`: Use cached dataset (default: True)
- `--fn-penalty`: False negative penalty factor (default: 2.0)

### Testing

Test a trained model:

```bash
python main.py test --checkpoint checkpoints/sl_binary_classifier_best.pth
```

### Preprocessing

Preprocess SL dataset (extract frames):

```bash
python main.py preprocess-sl --extract-frames --videos-dir data/SL_videos --output-dir data/SL_frames_output
```

Preprocess Jester dataset:

```bash
python main.py preprocess-jester --jester-csv data/Train.csv --jester-train-path data/Train --output-path data/jester_2_classes
```

### Convert to Numpy Arrays

Convert videos to numpy arrays:

```bash
python main.py convert-videos --input-dir data/SL_retained_classes --output-dir data/SL_YES_npy
```

Convert frame folders to numpy arrays:

```bash
python main.py convert-frames --input-dir data/jester_2_classes --output-dir data/jester_NO_npy
```

## Model Architecture

### LSTM Classifier

- Bidirectional LSTM layers
- Attention mechanism over sequence outputs
- Fully connected classification head
- Input: (batch_size, 37, 126) - 37 frames, 126 landmark features
- Output: (batch_size, 2) - binary classification logits

### Transformer Classifier

- Multi-head self-attention encoder
- Positional encoding
- Mean pooling over sequence
- Fully connected classification head
- Input: (batch_size, 37, 126)
- Output: (batch_size, 2)

## Configuration

Edit `config/config.yaml` to customize:

- Model architecture (lstm/transformer)
- Training hyperparameters (batch size, learning rate, epochs)
- Data splits and augmentation
- Device settings (CUDA, mixed precision)
- Logging and checkpointing

Key parameters:

```yaml
model:
  model_type: 'lstm'
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

## Training Details

- Uses weighted cross-entropy loss with false negative penalty
- Saves best model based on validation YES recall (not accuracy)
- Supports multiple optimizers (Adam, SGD, AdamW)
- Supports multiple schedulers (Cosine, Step, ReduceLROnPlateau)
- Gradient clipping for training stability
- Early stopping based on validation recall

## Data Format

Each sample is a numpy array of shape `(37, 126)`:
- 37 frames (sequence length)
- 126 features = 2 hands × 21 landmarks × 3 coordinates (x, y, z)

Dataset CSV format:
```
filepath,class,label
data/SL_YES_npy/video1.npy,yes,1
data/jester_NO_npy/video2.npy,no,0
```

## Performance Metrics

The model prioritizes:
- YES class recall (minimize false negatives)
- Overall accuracy
- Precision-recall balance

Training outputs confusion matrix and per-class metrics.
