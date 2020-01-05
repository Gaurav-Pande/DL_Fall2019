import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        # kernel_size = 3
        # print(im_size, hidden_dim, kernel_size, n_classes)
        self.conv = nn.Conv2d(im_size[0], hidden_dim, kernel_size)
        output_size = im_size[1] - kernel_size + 1 # padding = 0, stride = 1
        self.linear_layer = nn.Linear(hidden_dim * output_size * output_size, n_classes)
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = self.conv(images)
        relu_here = nn.ReLU()
        scores = relu_here(scores)
        scores = self.linear_layer(scores.reshape((images.shape[0], -1)))
        return scores