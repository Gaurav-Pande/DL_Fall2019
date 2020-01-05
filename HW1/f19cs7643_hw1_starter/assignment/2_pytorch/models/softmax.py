import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        self.channels, self.height, self.width = im_size
        self.input_dimensions = self.channels * self.height * self.width
        self.output_dimensions = n_classes
        self.linear_layer = nn.Linear(self.input_dimensions, n_classes, bias=True)
        
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        scores = None
        N, C, H, W = images.shape
        images_in = torch.reshape(images, (N, -1))
        scores = self.linear_layer(images_in) # N x num_classes
        return scores
        # softmax = nn.Softmax(dim = 1)
        # softmax_out = softmax(linear_out)
        # return softmax_out
        # return linear_out
        

        #############################################################################
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        # return scores

