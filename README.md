# Reproducing SiNSR: Experiments under Limited and Full GPU Settings

**SinSR: Diffusion-Based Image Super-Resolution in a Single Step**

This repository is based on the official implementation of the paper  ["SinSR: Diffusion-Based Image Super-Resolution in a Single Step"](https://arxiv.org/pdf/2311.14760.pdf).

![Alt text](assets/framework.png)
## Overview
This repository is based on the official implementation of **SiNSR**.
The goal of this project is to reproduce the SinSR model and verify its behavior
under both limited computational resources and the original experimental settings
described in the paper.

## Experimental Setup and Reproduction Strategy
To validate the functionality and behavior of the SinSR model, 
experiments were conducted in two stages.

**Stage 1: Reduced-scale verification**
Due to limited GPU resources, an initial experiment was performed
to confirm the core behavior of the SinSR model.
The training setup was intentionally simplified by reducing the dataset size
to approximately 5,000 images, setting the batch size to 1, and lowering
the image resolution to 128.
This stage aimed to verify that the model functions as described in the paper
before scaling up to the full experimental setting.

**Stage 2: Full-scale reproduction**
After confirming the model behavior, the experiment was repeated
under conditions matching the original paper.
Using two GPUs, the model was trained with approximately 70,000 training images,
image resolution of 256, batch size of 4, and 500,000 training iterations.
This stage focuses on reproducing the quantitative and qualitative results
reported in the original SinSR paper.

