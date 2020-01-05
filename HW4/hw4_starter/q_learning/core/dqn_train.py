import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from core.q_train import QNTrain


class DQNTrain(QNTrain):
    """
    Class for training a DQN
    """
    def __init__(
        self,
        q_net_class,
        env,
        config,
        device,
        logger=None,
    ):
        super().__init__(env, config, logger)
        self.device = device

        self.q_net = q_net_class(env, config)
        self.target_q_net = q_net_class(env, config)

        self.q_net.to(device)
        self.target_q_net.to(device)

        self.update_target_params()
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        self.target_q_net.eval()

        if config.optim_type == 'adam':
            self.optimizer = optim.Adam(
                self.q_net.parameters(), lr=config.lr_begin)
        elif config.optim_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.q_net.parameters(), lr=config.lr_begin)
        else:
            raise ValueError(f"Unknown optim_type: {config.optim_type}")


    def process_state(self, state):
        """
        Processing of state

        Args:
            state: np.ndarray of shape either (batch_size, H, W, C)
            or (H, W, C), of dtype 'np.uint8'

        Returns:
            state: A torch float tensor on self.device of shape
            (*, H, W, C), where * = batch_size if it was present in
            input, 1 otherwise. State is normalized by dividing by
            self.config.high
        """
        #####################################################################
        # TODO: Process state to match the return output specified above.
        #####################################################################
        #print("process state")
        #print(len(state.shape))
        state = torch.from_numpy(state/self.config.high).float()
        #print("gamma", self.config.gamma)
        #state = state/self.config.high
        if len(state.shape) == 3:
            state.unsqueeze_(0)
        assert len(state.shape) == 4
        state = state.to(device=self.device)
        #print("state")
        #print(state.shape)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return state


    def forward_loss(
        self,
        state,
        action,
        reward,
        next_state,
        done_mask,
    ):
        """
        Compute loss for a batch of transitions. Transitions are defiend as
        tuples of (state, action, reward, next_state, done).

        Args:
            state: batch of states (batch_size, *)
            action: batch of actions (batch_size, num_actions)
            next_state: batch of next states (batch_size, *)
            reward: batch of rewards (batch_size)
            done_mask: batch of boolean values, 1 if next state is terminal
                state (ending the episode) and 0 otherwise.

        Returns:
            The loss for a transition is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2

            Notation:
                s, s': current and next state respectively
                a: current action
                a': possible future actions at s'
                Q: self.q_net
                Q_target: self.target_q_net
        """
        #####################################################################
        # TODO: Compute the loss defined above by using the Q networks
        #   self.q_net and self.target_q_net
        #
        # HINT: The output of the Q network is a tensor of state-action
        #       values for ALL actions, but Q(s, a) denotes the state-action
        #       values for ONLY the actions taken by the agent. With the
        #       help of torch.gather, Q(s, a) can be computed given the
        #       batch of current states and actions.
        #
        #       After computing the Q(s, a) tensor of shape (batch_size),
        #       use the done_mask to compute Q_samp(s) from the equation
        #       specified above.
        #####################################################################
#         print("calculating forward loss")
#         q_samp = torch.where(done_mask, reward, reward + config.gamma * torch.max(self.target_q_net, axis=1))
#         q_expected = self.q_net(state).gather(1, action)
#         loss= (q_samp-q_expected) ** 2
        q_sample = reward.clone()
        q1 = self.q_net(state)
        q_a = torch.gather(q1,1,action.view(-1,1)).squeeze(1)
        #print(len(q_sample))
        done_mask = done_mask.bool()
        done_mask = (~done_mask)
        done_mask = done_mask.float()
        #n_done_mask = []
#         for mask in done_mask:
#             n_done_mask.append(float(~mask))
        #done_mask = float(~done_mask)
        #print(n_done_mask)
        #print(len(self.config.gamma*self.target_q_net(next_state).max(1).values))
#         for i,v in enumerate(done_mask):
#             if not done_mask[i]:
#                 q_sample[i] += self.config.gamma*self.target_q_net(next_state).max(1).values    
        q_sample += (done_mask)*self.config.gamma*self.target_q_net(next_state).max(1).values
        loss = ((q_sample - q_a)**2).sum()
        
        
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return loss


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        
        #####################################################################
        # TODO: Update the parameters of self.target_q_net with the
        # parameters of self.q_net, refer to the documentation of
        # torch.nn.Module.load_state_dict and torch.nn.Module.state_dict
        # This should just take 1-2 lines of code.
        #####################################################################
#         for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
#             target_param.data.copy_(local_param.data)
#         return
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################


    def module_grad_norm(self, net):
        """
        Compute the L2 norm of gradients accumulated in net
        """
        with torch.no_grad():
            total_norm = 0
            for param in net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            return total_norm


    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        torch.save(self.q_net.state_dict(),
            os.path.join(self.config.model_output,
                f"{self.q_net.__class__.__name__}.vd"))


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        state = self.process_state(state)
        action_values = self.q_net(state)

        return np.argmax(action_values.cpu().numpy()), action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate

        Returns:
            q_loss: Loss computed using self.forward_loss
            grad_norm_eval: L2 norm of self.q_net gradients, computed
                using self.module_grad_norm
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch \
            = replay_buffer.sample(self.config.batch_size)

        #####################################################################
        # TODOs:
        # 1. Process all the arrays sampled from the replay buffer
        #   states -> use self.process_state
        #   everything else -> convert np.ndarrays to torch tensors
        #   and move them to self.device
        #   NOTE: actions must be converted to long tensors using .long()
        #
        # 2. After you implement self.forward_loss, use it here with
        #   the processed tensors as input to compute q_loss.
        #
        # 3. Update the Q network with one step of self.optimizer after
        #   calling backward on q_loss. Make sure to compute the L2 norm
        #   of the gradients using self.module_grad_norm on self.q_net
        #   AFTER calling backward.
        #####################################################################
        s_batch = self.process_state(s_batch)
        # if there are any batched data check using the len of the shape
        assert len(s_batch.shape) == 4
        sp_batch = self.process_state(sp_batch)
        assert len(sp_batch.shape) == 4
        a_batch = torch.from_numpy(a_batch).long().to(device = self.device)
        r_batch = torch.from_numpy(r_batch).to(device = self.device)
        done_mask_batch = torch.from_numpy(done_mask_batch).to(device = self.device)
        
        
        q_loss = self.forward_loss(s_batch,a_batch,r_batch,sp_batch,done_mask_batch)
        self.optimizer.zero_grad()
        q_loss.backward()
        #l2 norm calculation
        grad_norm_eval = self.module_grad_norm(self.q_net)
        self.optimizer.step()
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
        return q_loss, grad_norm_eval
