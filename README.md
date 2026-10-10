# CS5720-Home-Assignment-3

**Course:** CS5720 Neural Networks and Deep Learning, Fall 2026
**University:** University of Central Missouri
**Student Name:** Anuradha Kurupathi
**Student ID:** 700781806

## Overview
This repository contains my solutions for Home Assignment 3. Part I covers short-answer questions on convolution, padding and stride, CNN architectures, and representation/transfer learning. Part II covers a from-scratch 2-D convolution implementation in NumPy (Q1) and a transfer learning experiment comparing a frozen feature extractor with a fine-tuned network (Q2).

## Repository Contents
- `part2_q1_convolution.py` — 2-D convolution from scratch using NumPy
- `part2_q2_transfer_learning.py` — ResNet18 frozen vs. fine-tuned experiment
- `training_loss.png` — training loss plot for both experiments
- `README.md` — this file

---

# Part I: Short-Answer Questions

## Q1: Convolution, Padding, and Stride
- Input 32×32×3, 16 filters of size 5×5, stride S = 1, padding P = 0
- **a.** N = 32, F = 5, S = 1, P = 0
- **b.** N_out = (N − F + 2P)/S + 1 = (32 − 5 + 0)/1 + 1 = 28, so the feature map is 28×28
- **c.** One output channel per filter, so the output volume is 28×28×16
- **d.** To keep 32×32: (32 − 5 + 2P)/1 + 1 = 32, so 2P = 4 and **P = 2**

## Q2: CNN Architectures
- **AlexNet:** Showed that a deep CNN trained on GPUs, with ReLU, dropout, and data augmentation, could far outperform traditional methods on ImageNet (2012).
- **VGGNet:** Stacks small 3×3 filters instead of large ones. Two 3×3 layers cover the same receptive field as one 5×5, and three cover a 7×7, but with fewer parameters (27C² vs. 49C²) and more non-linearities.
- **GoogLeNet (Inception):** Applies 1×1, 3×3, and 5×5 convolutions and pooling in parallel, then concatenates the results to capture features at multiple scales. 1×1 convolutions reduce channels first to keep computation low.
- **ResNet:** Residual (skip) connections learn F(x) = H(x) − x and output F(x) + x. This addresses vanishing gradients and the degradation problem, so very deep networks can be trained.

## Q3: Representation Learning and Transfer Learning
- **Representation learning:** The model automatically learns useful features from raw data instead of using hand-crafted features.
- **Lower vs. higher layers:** Lower layers have small receptive fields and learn generic patterns (edges, colors, textures). Higher layers combine them into abstract, task-specific features (object parts, whole objects).
- **Transfer learning / fine-tuning:** Transfer learning reuses a model pretrained on a large dataset for a new task. Fine-tuning further trains some or all of the pretrained weights on the new data, usually with a small learning rate.
- **Why pretrained on a small dataset:** Training a deep CNN from scratch on little data overfits and is expensive. A pretrained CNN already has general features, so it needs less data, trains faster, and generalizes better.
- **Freezing vs. fine-tuning:** A frozen layer has fixed weights (no gradient updates). A fine-tuned layer is unfrozen and its weights are updated during training.

---

# Part II: Programming Questions

## Q1: Implementing Convolution from Scratch
- Stored the 5×5 input and 3×3 filter `[[1,0,1],[0,1,0],[1,0,1]]` as NumPy arrays
- Wrote `conv2d()` that slides the filter over the input and computes the dot product (element-wise multiply, then sum) at every valid location, with no built-in convolution function
- Used stride = 1 and padding = 0
- Output: complete feature map and its shape

Result:

| Setting | Output shape | Output |
| --- | --- | --- |
| Stride 1, padding 0 | 3×3 | [[4, 3, 4], [2, 4, 3], [2, 3, 4]] |
| Stride 2, padding 0 | 2×2 | [[4, 4], [2, 4]] |

Effect of stride: With stride 2 the filter moves 2 pixels at a time, so it visits fewer positions. Output size becomes ⌊(5 − 3)/2⌋ + 1 = 2. A larger stride downsamples the feature map and reduces computation, but skips positions and loses some fine spatial detail.

## Q2: Transfer Learning, Freeze vs. Fine-Tune
- Loaded an ImageNet-pretrained **ResNet18** (torchvision)
- Dataset: CIFAR-10 subset with 2 classes (airplane vs. ship), 2,000 training images and 1,000 test images, resized to 112×112 and normalized with ImageNet statistics
- Both experiments used the same seed, 5 epochs, batch size 64, and Adam (lr = 1e-3)
- **Experiment A (Feature Extraction):** froze all convolutional/base layers, replaced the final layer with a 2-class `Linear` layer, and trained only that layer
- **Experiment B (Fine-Tuning):** froze everything except the last convolutional block (`layer4`), and trained `layer4` plus the new classifier
- Recorded trainable parameters, training time, and test accuracy, and plotted the training loss for both
- Output: training log, `training_loss.png`, and comparison table

Results:

| Method | Trainable Parameters | Training Time | Accuracy |
| --- | --- | --- | --- |
| Frozen Feature Extractor | 1,026 | 12.7 s | 86.50% |
| Fine-Tuned Network | 8,394,754 | 10.9 s | 94.10% |

![Training loss](training_loss.png)

Discussion:
Fine-tuning reached 94.10% test accuracy compared with 86.50% for the frozen feature extractor, an improvement of 7.6 percentage points. The frozen model keeps the ImageNet features unchanged and only learns a new classifier on top, while fine-tuning lets the last convolutional block adapt its high-level features to airplane and ship images. The fine-tuned model trains 8,394,754 parameters versus only 1,026, so in theory it should take longer per epoch, since gradients must flow back through `layer4` and its weights must be updated. In my run the fine-tuned model was slightly faster (10.9 s vs. 12.7 s), which I believe comes from one-time overhead (GPU initialization, data caching) during the first, frozen run, and from data loading taking a large share of each epoch on a small dataset. In general, freezing layers reduces memory use and backward-pass computation and lowers the risk of overfitting, while fine-tuning costs more compute but usually gives better accuracy when the new data differs from ImageNet. For this task, the extra trainable parameters were clearly worth it.

## How to Run
1. Install the requirements: `pip install numpy torch torchvision matplotlib`
2. Run `python part2_q1_convolution.py` for the convolution output.
3. Run `python part2_q2_transfer_learning.py` for the transfer learning experiments (downloads CIFAR-10 and the pretrained ResNet18 weights on first run; a GPU is recommended, e.g. a T4 in Google Colab).
