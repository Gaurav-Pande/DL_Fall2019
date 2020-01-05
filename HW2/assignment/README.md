---
layout:     page
title:      Homework 2, Question 6
permalink:  /hw2/
---

# [CS 7643 Deep Learning - Homework 2][1]

In this homework, we will learn different ways of visualizing and using data gradients, including saliency maps, fooling images, class visualizations, and style transfer. This homework is divided into two parts:

- Understand network visualization and implement saliency maps, fooling images, class visualizations
- Understand and implement style transfer

Note that this homework is [assignment 3 from the Standford CS231n course][2].

Download the starter code [here]({{site.baseurl}}/assets/f19cs7643_hw2_starter.zip).

## Setup

Assuming you already have homework 1 dependencies installed, here is some prep work you need to do. First, install the `future` package:

```
pip install future
```

Then download the `imagenet_val_25` dataset

```
cd cs7643/datasets
./get_imagenet_val.sh
```

We will use PyTorch 1.1+  to complete the problems in this homework, and has been tested with Python 3.6+ on Linux and Mac. Note that the future package is used to provide some backwards compatibility with lower versions of Python. If you are using Python 3+, you may need to use `pip3 install future` instead.

Throughout this homework, we will use [SqueezeNet][7], which should enable you to easily perform all the experiments on a CPU. You are encouraged to use a larger model to finish the rest of the experiments if GPU resouces are not a problem for you, but please highlight the backbone network you use in your implementation if you do it.

Switching a backbone network is quite easy in PyTorch. You can refer to the [torch.vision model zoo][6] for more information.

* [Iandola et al, "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5MB model size", arXiv 2016][7]

## Part 1

Open notebook `NetworkVisualization-Pytorch.ipynb`. We will explore the use of *image gradients* for generating new images, by studying and implementing key components in three papers:

1. [Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR Workshop 2014.][3]
2. [Szegedy et al, "Intriguing properties of neural networks", ICLR 2014][4]
3. [Yosinski et al, "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning Workshop][5]

You will need to first read these papers, and then we will guide you to understand them deeper with some problems.

### Q1.1: Saliency Maps (10 points)

You need to implement the `compute_saliency_maps` function referring to section 3 of the [first paper][3], which describes a method to understand which part of an image is important for classification by visualizing the gradient of the correct class score with respect to the input image. You first want to compute the loss over the correct scores, and then compute the gradients with a backward pass.

### Q1.2: Generating Fooling Images (10 points)

Several papers have suggested ways to perform optimization over the input image to construct images that break a trained ConvNet. Given a trained ConvNet, an input image, and a desired label, we can add a small amount of noise to the input image to force the ConvNet to classify it as having the desired label. You need to generate a fooling image in `make_fool_image` referring to the [second paper][4]. You should perform gradient ascent on the score of the target class, stopping when the model is fooled.

### Q1.3 Class Visualization (10 points) [Extra Credit for CS4803]

You need to implement the `create_class_visualization` function referring to the [third paper][5]. By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class.

**Deliverables**

Submit the notebook you finished with all the generated outputs. (Note: This is still extra credit for CS4803 even though the starter code doesn't indicate that where it assigns a point value to this question.)

Just as in HW1, you will need to use `nbconvert` or the 'Download as PDF' option in Jupyter to convert the notebook into PDFs, with all of your changes.

## Part 2

Another application of image gradients is style transfer. This has recently become quite popular. In this notebook, we will study and implement the style transfer technique from:

* [Gatys et al., "Image Style Transfer Using Convolutional Neural Networks", CVPR 2015][8].

The general idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic style of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep network, and then performing gradient descent on the pixels of the image itself.

Open notebook `StyleTransfer-Pytorch.ipynb`. Implement the loss functions for this task and the training update code.

### Q2.1 Implement Content Loss (3 points)

Content loss measures how much the feature map of the generated image differs from the feature map of the source image. Implement the `content_loss` function and pass the `content_loss_test`.

### Q2.2 Implement Style Loss (6 points)

First, compute the Gram matrix which represents the correlations between the responses of each filter, by implementing the function `gram_matrix` and pass `gram_matrix_test`. Then implement `style_loss` function and pass the `style_loss_test`. Each of the function worth 3 points.

### Q2.3 Implement Total Variation Loss (3 points)

Implement total variation regularization loss in `tv_loss`, which is the sum of the squares of differences in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically). You need to both pass `tv_loss_test` and provide an efficient vectorized implementation to receive the full credit.

### Q2.4 Finish Style Transfer (6 points)

Read the `style_transfer` function and figure out what are all the parameters, inputs, solvers, etc. The update rule in the following block is hold out for you to finish. What you need to implement is the update rule with by forwarding it to criterion functions and perform the backward update.  

You need to generate the pretty pictures outputs which are similar to the given examples in the following block to receive full credits.

### Q2.5 Feature Inversion (2 points)

Suppose you implement things correctly, what you have done can do another cool thing. In an attempt to understand the types of features that convolutional networks learn to recognize, the following paper attempts to reconstruct an image from its feature representation. We can easily implement this idea using image gradients from the pretrained network, which is exactly what we did above (but with two different feature representations).

* [Aravindh Mahendran, Andrea Vedaldi, "Understanding Deep Image Representations by Inverting them", CVPR 2015][9]

Just run this block and generate the outputs. If you previous implementation is correct, you will get the full credits.

**Deliverables**

Submit the notebook you finished with all the generated outputs.

Just as in HW1, you will need to use `nbconvert` or the 'Download as PDF' option in Jupyter to convert the notebook into PDFs, with all relevant results.

For each of the loss function in part 2, **you will need to pass the unit test to receive full credit, otherwise it will get a score of 0.** For the final output you will be expected to generate the images similar to the output to receive the full credits.  


## Submit your homework
First, combine all of your PDFs into one PDF, in the following order:

1. Your solutions to questions in PS2
2. Your NetworkVisualization converted PDF
3. Your StyleTransfer converted PDF

This PDF will be submitted under the HW2 designation in Gradescope.

Run `collect_submission.sh`

```
./collect_submission.sh
```

which should generate the off-the-shelf runnable notebook with all of your implementations in a ZIP file, as well as your PDFs.
Submit this ZIP to the HW2 Code designation in Gradescope.

Although we will run your notebook in grading, you still need to **submit the notebook with all the outputs you generated**. Sometimes it will inform us if we get any inconsistent results with respect to yours.

References:

1. [CS231n Convolutional Neural Networks for Visual Recognition][2]

[1]: https://www.cc.gatech.edu/classes/AY2018/cs7643_fall/
[2]: http://cs231n.github.io/assignments2017/assignment3/
[3]: https://arxiv.org/abs/1312.6034
[4]: https://arxiv.org/abs/1312.6199
[5]: http://yosinski.com/deepvis
[6]: https://github.com/pytorch/vision#models
[7]: https://arxiv.org/abs/1602.07360
[8]: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
[9]: https://arxiv.org/abs/1412.0035
