# CNTK Examples: Image/Classification/NIN

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define Network in Network (NIN) (https://arxiv.org/abs/1312.4400) for image classification.
|Network   |Network in Network.
|Training  |
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train the AlexNet which won the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2012 challenge. AlexNet is an important milestone, as for the first time it was shown that deep convolutional neural networks can outperform traditional manual feature design for vision tasks by a significant margin.

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

To be added.

## Pre-trained Models

### Caffe-Converted

|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/NIN_ImageNet.model
|:---------|:---
|Source Caffe model website | https://gist.github.com/mavenlin/d802a5849de39225bcc6
|Single crop top 5 error | 18.93%
