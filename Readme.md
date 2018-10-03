# Textile Defect Detection by Yarn Tracking and Fully Convolutional Networks
In this repo, you can find our implementation leading to the paper:   
Defect Detection in Plain Weave Fabrics by Yarn Tracking and Fully Convolutional Networks:
https://www.lfb.rwth-aachen.de/bibtexupload/pdf/WEN18a.pdf

## Introduction

Based on a deep learning segmentation network and rule-based yarn tracking, defects can be found in unseen textiles.
For this purpose, the following consecutive steps were implemented:
- Identification of single weft and warp float-points with a fully convolutional neural network
- Tracking of yarns based on a set of rules
- Plausibility checking for yarn tracking (optional)
- Locating defects based on anomaly detection

The provided network can be used as it is for unseen tissues. However, to reduce false positives, 
we recommend retraining it on fabrics similar to the one where the defects should be detected.

## Prerequisites:

For running this code, we recommend to use a CUDA enabled graphics card with Cuda 9.0 and cuDNN > 7.2.  
Also, you'll need the following python packages:

Python 3.6  
Tensorflow 1.11  
OpenCV  
Numpy  
matplotlib  
Scipy

## Running

We provide a pre-trained network (example/net.h5), together with an exemplary fabric, which was not in the train set (example/orig_fabric).
When running this code, you should get the same results as in example/defects_detected. The results of the intermediate
steps are also provided.

When run on a complete fabric, i.e., on several hundred of images, we plot the statistics over all images. This step, 
as well as the training step was deactivated in this code example. Depending on your use case, these should also
be included.

If you have a problem running this code please open an issue.