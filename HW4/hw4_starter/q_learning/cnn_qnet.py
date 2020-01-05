import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H,W,C = env.observation_space.shape 
        self.first = nn.Conv2d(C*config.state_history,16,kernel_size = 8,stride = 4)
        self.relu_first = nn.ReLU()
        h1  = (H-8)//4 + 1
        self.second = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.relu_second = nn.ReLU()
        h_out = (h1-4)//2 + 1
        self.output_size = env.action_space.n
        self.fully_connected = nn.Linear(h_out*h_out*32,self.output_size)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        batch_size = len(state)
        state = state.transpose(1,3)
        o = self.first(state)
        o = self.relu_first(o)
        o=self.second(o)
        o=self.relu_second(o)
        o=o.reshape(batch_size,-1)
        o = self.fully_connected(o)
        return o
        
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
