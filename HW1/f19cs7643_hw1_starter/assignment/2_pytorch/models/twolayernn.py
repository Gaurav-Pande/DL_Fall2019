import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        channels, height, width = im_size
        self.input_dim = channels * height * width
        self.first_layer = nn.Linear(self.input_dim, hidden_dim, bias=True)
        self.second_layer = nn.Linear(hidden_dim, n_classes, bias=True)
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
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
        # scores = None
        N, C, H, W = images.shape
        images_input = torch.reshape(images, (N, -1))
        first_layer_out = self.first_layer(images_input)
        relu = nn.ReLU()
        relu_out = relu(first_layer_out)
        second_layer_out = self.second_layer(relu_out) # N x n_classes
        softmax = nn.Softmax(dim = 1)
        softmax_out = softmax(second_layer_out)
        return softmax_out

        #############################################################################
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        # return scores+

