# Training Guide

This guide explains how to train gaze prediction models using the training script.

## Pre-Training Setup

Before starting any training run, follow these steps:

### 1. Pull Required Resources

Use the Makefile to download and prepare datasets:

#### Pull Dataset Using Makefile

```bash
# Pull dataset from remote server
make pull SERVER=https://your-server-url

# This will:
# 1. Download remote data via download-remote.py
# 2. Preprocess data via preprocess-remote.py
# 3. Create test dataset via create-test-dataset.py
```

**Required:**
- `SERVER` environment variable or command-line argument
- Network access to the remote server
- Sufficient disk space for dataset

### 2. Run Training

### Training a Teacher Model

```bash
python scripts/train.py \
  --batch-size 32 \
  --epochs 250 \
  --patience 5 \
  --model latest \
  --mode full \
  --sampling oversample \
  --teacher-scale 2 \
  --trainer teacher \

```

### Training a Student Model with Knowledge Distillation

```bash
python scripts/train.py \
  --batch-size 32 \
  --epochs 250 \
  --patience 5 \
  --mode full \
  --sampling oversample \
  --teacher-scale 2 \
  --trainer student \
  --teacher-weights best \
  --feature-match-weight 0.5
```
