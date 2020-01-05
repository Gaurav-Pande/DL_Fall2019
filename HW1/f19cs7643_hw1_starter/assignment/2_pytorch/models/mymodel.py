import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()

        self.conv = nn.Conv2d(im_size[0], hidden_dim, kernel_size, padding = (kernel_size - 1) // 2) 
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding = (kernel_size - 1) // 2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # size = 16
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding = (kernel_size - 1) // 2)
        self.linear_layer = nn.Linear(hidden_dim * 16 * 16, hidden_dim * 16 * 16 // 8)
        self.linear_layer2 = nn.Linear(hidden_dim * 16 * 16 // 8, hidden_dim * 16 * 16 // 32)
        self.linear_layer3 = nn.Linear(hidden_dim * 16 * 16 // 32, n_classes)

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        scores = self.conv2(scores)
        relu_here = nn.ReLU()
        scores = relu_here(scores)
        scores = self.pool(scores)
        scores = self.conv3(scores)
        relu_here = nn.ReLU()
        scores = relu_here(scores)
        scores = self.linear_layer(scores.reshape((images.shape[0], -1)))
        relu_here = nn.ReLU()
        scores = relu_here(scores)
        scores = self.linear_layer2(scores)
        relu_here = nn.ReLU()
        scores = relu_here(scores)
        scores = self.linear_layer3(scores)
        return scores

# class MyModel(nn.Module):
#     def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
#         '''
#         Extra credit model

#         Arguments:
#             im_size (tuple): A tuple of ints with (channels, height, width)
#             hidden_dim (int): Number of hidden activations to use
#             kernel_size (int): Width and height of (square) convolution filters
#             n_classes (int): Number of classes to score
#         '''
#         super(MyModel, self).__init__()
#         # self.conv_layer = self.create_conv_layer()
#         self.conv_layer = nn.Sequential (

#             nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace = True),

#             nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 64 * 16 * 16

#             nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace = True),

#             nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 256 * 8 * 8

#             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 256 * 4 * 4
#         )
#         # self.linear_layer = self.create_linear_layer()
#         self.linear_layer = nn.Sequential (
#             nn.Dropout(p=0.2),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace = True),
#             nn.Dropout(p=0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace = True),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 10),
#             nn.ReLU(inplace = True),
#         )

#     def create_conv_layer(self):
#         return nn.Sequential (

#             nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace = True),

#             nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 64 * 16 * 16

#             nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace = True),

#             nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 256 * 8 * 8

#             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace = True),

#             nn.MaxPool2d(kernel_size = 2, stride = 2), # 256 * 4 * 4
#         )
    
#     def create_linear_layer(self):
#         return nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace = True),
#             nn.Dropout(p=0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace = True),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 10),
#             nn.ReLU(inplace = True),
#         )

#     def forward(self, images):
#         '''
#         Take a batch of images and run them through the model to
#         produce a score for each class.

#         Arguments:
#             images (Variable): A tensor of size (N, C, H, W) where
#                 N is the batch size
#                 C is the number of channels
#                 H is the image height
#                 W is the image width

#         Returns:
#             A torch Variable of size (N, n_classes) specifying the score
#             for each example and category.
#         '''
#         scores = self.conv_layer(images)
#         scores = scores.view(scores.size(0), -1)
#         scores = self.linear_layer(scores)
#         return scores

# vgg16 = [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'] 
# vgg16 = [64, 64, 'P', 128, 128, 'P', 256, 256, 'P'] 

# class MyModel(nn.Module):
#     def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
#         '''
#         Extra credit model

#         Arguments:
#             im_size (tuple): A tuple of ints with (channels, height, width)
#             hidden_dim (int): Number of hidden activations to use
#             kernel_size (int): Width and height of (square) convolution filters
#             n_classes (int): Number of classes to score
#         '''
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(im_size[0], 64, kernel_size = 3, padding = 1)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
#         self.batch_norm2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
#         self.batch_norm3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
#         self.batch_norm4 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
#         self.linear1 = nn.Linear(8192, 512)
#         self.linear2 = nn.Linear(512, 10)

#     def forward(self, images):
#         '''
#         Take a batch of images and run them through the model to
#         produce a score for each class.

#         Arguments:
#             images (Variable): A tensor of size (N, C, H, W) where
#                 N is the batch size
#                 C is the number of channels
#                 H is the image height
#                 W is the image width

#         Returns:
#             A torch Variable of size (N, n_classes) specifying the score
#             for each example and category.
#         '''
#         scores = self.conv1(images)
#         scores = self.batch_norm1(scores)
#         relu1 = nn.ReLU()
#         scores = relu1(scores)
#         #print("Shape ", scores.shape)

#         scores = self.conv2(scores)
#         scores = self.batch_norm2(scores)
#         relu2 = nn.ReLU()
#         scores = relu2(scores)
#         #print("Shape ", scores.shape)

#         scores = self.pool1(scores)
#         #print("Shape ", scores.shape)

#         scores = self.conv3(scores)
#         scores = self.batch_norm3(scores)
#         relu3 = nn.ReLU()
#         scores = relu3(scores)
#         #print("Shape ", scores.shape)

#         scores = self.conv4(scores)
#         scores = self.batch_norm4(scores)
#         relu4 = nn.ReLU()
#         scores = relu4(scores)
#         #print("Shape ", scores.shape)

#         scores = self.pool2(scores)
#         #print("Shape ", scores.shape)

#         relu5 = nn.ReLU()
#         scores = relu5(scores)
#         #print("Shape ", scores.shape)

#         scores = self.linear1(scores.reshape((images.shape[0], -1)))
#         relu6 = nn.ReLU()
#         scores = relu6(scores)
#         #print("Shape ", scores.shape)

#         scores = self.linear2(scores)
#         relu7 = nn.ReLU()
#         scores = relu7(scores)
#         #print("Shape ", scores.shape)

#         return scores

# class MyModel(nn.Module):
#     def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
#         '''
#         Extra credit model

#         Arguments:
#             im_size (tuple): A tuple of ints with (channels, height, width)
#             hidden_dim (int): Number of hidden activations to use
#             kernel_size (int): Width and height of (square) convolution filters
#             n_classes (int): Number of classes to score
#         '''
#         super(MyModel, self).__init__()
#         self.conv_model = self.make_model(vgg16)
#         ##print(self.conv_model)
#         self.linear_layer = nn.Linear(4096, 2048)
#         self.relu = nn.ReLU()
#         self.linear_layer2 = nn.Linear(2048, 512)
#         self.relu2 = nn.ReLU()
#         self.linear_layer3 = nn.Linear(512, n_classes)

#     def make_model(self, model_list):
#         '''
#         Arguments:
#             model_list (tuple): A tuple with list of layers in the model
        
#         Returns:
#             A constructed model as per the layers
#         '''
#         layers = []
#         in_channels = 3
#         for layer_type in model_list:
#             if layer_type == 'P':
#                 layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2)) #Fixed for VGG
#             else:
#                 out_channels = layer_type
#                 layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
#                 layers.append(nn.ReLU())
#                 in_channels = out_channels
#         return nn.Sequential(*layers)

#     def forward(self, images):
#         '''
#         Take a batch of images and run them through the model to
#         produce a score for each class.

#         Arguments:
#             images (Variable): A tensor of size (N, C, H, W) where
#                 N is the batch size
#                 C is the number of channels
#                 H is the image height
#                 W is the image width

#         Returns:
#             A torch Variable of size (N, n_classes) specifying the score
#             for each example and category.
#         '''
#         scores = self.conv_model(images)
#         scores = self.linear_layer(scores.reshape((images.shape[0], -1)))
#         scores = self.relu(scores)
#         scores = self.linear_layer2(scores)
#         scores = self.relu(scores)
#         scores = self.linear_layer3(scores)
#         return scores