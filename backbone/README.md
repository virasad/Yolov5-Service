- [Dataset](#dataset)
  - [Before Augment](#before-augment)
  - [After Augment](#after-augment)
    - [Train](#train)
    - [Validation](#validation)
- [Train](#train-1)
  - [Network method](#network-method)
- [Detect System](#detect-system)
- [Train network method](#train-network-method)
  - [Input Image](#input-image)
  - [Learning params](#learning-params)
- [Result](#result)
  - [Train Result](#train-result)
  - [confusion matrix](#confusion-matrix)
  - [F1 Score](#f1-score)
  - [Percision score](#percision-score)
  - [Recall score](#recall-score)
  - [Persition/Recall score](#persitionrecall-score)
  - [Sample result in validation](#sample-result-in-validation)
- [How to run](#how-to-run)
- [Setting Up docker GPU](#setting-up-docker-gpu)
# Dataset 
## Before Augment
32 images

## After Augment
5000

### Train
4000

### Validation
1000

# Train
## Network method
YOLOv5s 

# Detect System

OS: 20.04.1-Ubuntu
Kernel: 5.11.0-27-generic
CPU: Intel(R) Core(TM) i7-6700K CPU @ 4
RAM: 19GiB DDR4
Graphic: GeForce GTX 960

# Train network method

YOLOv5s6

**Why we choose this network**
We need both accuracy and focus on detailed so we use this network since the input image size is large enough in order to detect small objects and simultaneously has both speed and accuracy.

## Input Image
1280*1280

## Learning params
```
lr0: 0.01
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
epochs: 300
```

# Result
## Train Result
![Train result](results/results.png)

## confusion matrix
![Confusion matrix](results/confusion_matrix.png )

## F1 Score
![F1 Score](results/F1_curve.png)

## Percision score
![Percision score](results/P_curve.png )

## Recall score
![Recall score](results/R_curve.png )

## Persition/Recall score
![Recall score](results/PR_curve.png )

## Sample result in validation
![Sample result](results/val_batch0_pred.jpg)


# How to run
1. First of all install Pytorch regarding to your os and your system config [here](https://pytorch.org/)

2. Config your data and network in `config/config.py`

3. `python detect.py`

# Setting Up docker GPU
[Here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
# `docker compose up`