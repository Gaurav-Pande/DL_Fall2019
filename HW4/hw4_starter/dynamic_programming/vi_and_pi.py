### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, value_function, gamma=0.9, tol=1e-3, DEBUG=False):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    #value_function = np.zeros(nS)
    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    running = True
    while running:
        val_func_new = np.zeros([nS])
        for state in range(nS):
            action = policy[state]
            for prob,next_s,reward,terminal in P[state][action]:
                val_func_new[state] += prob*(reward + gamma*value_function[next_s])
        val_change = np.max(np.abs(value_function-val_func_new))
        value_function = val_func_new
        if val_change <tol:
            running = False
            
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    #new_policy = np.zeros(nS, dtype='int')
    new_policy = np.copy(policy)
    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    from random import choice
    for state in range(nS):
        #action_reward = []
        q= np.zeros([nA])
        for action in range(nA):
            for prob,next_s,reward,terminal in P[state][action]:
            #prob,next_s,reward,terminal = P[state][action][0]
            #action_reward.append(reward + gamma*prob*value_from_policy[next_s])
                q[action] += prob*(reward + gamma*value_from_policy[next_s])
        new_policy[state] = choice(np.argwhere(q==q.max()))
        
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    running = True
    while running:
        n_value_function = policy_evaluation(P,nS,nA,policy,value_function,gamma,tol)
        n_pol = policy_improvement(P,nS,nA,value_function,policy,gamma)
        #pol_change= (n_pol!=policy).sum()
        #policy = n_pol
        if np.all(n_pol == policy):
            running = False
        
        policy = n_pol.copy()
        value_function = n_value_function.copy()
        
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    running = True
    while running:
        value_iter_new = np.copy(value_function)
        max_delta = 0
        for state in range(nS):
            #action_reward = []
            v_max = value_function[state]
            optimal_action = -1
            #max_delta = 0
            for action in range(nA):
                value_curr = 0
                for prob,next_s,reward,terminal in P[state][action]:
                    # for stochastic uncomment the below
#                     if reward == 0:
#                         reward = -10
#                     if reward == 1:
#                         reward = 10000000
                    value_curr += prob*(reward + gamma*value_iter_new[next_s])
                if value_curr > v_max:
                    v_max = value_curr
                    optimal_action = action
                    policy[state] = action
                #prob,next_s,reward,terminal = P[state][action][0]
                #action_reward.append(reward + gamma*prob*value_iter_new[next_s])
            value_iter_new[state]= v_max
            max_delta = max(max_delta,v_max-value_function[state])
        #loss=np.sum(np.abs(value_function-value_iter_new))
        value_function = value_iter_new
        if max_delta<tol:
            running = False
            
            
#     for state in range(nS):
#         action_reward = []
#         for action in range(nA):
#             prob,next_s,reward,terminal = P[state][action][0]
#             action_reward.append(reward + gamma*prob*value_iter_new[next_s])
#         policy[state]=np.argmax(action_reward)
        
       
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def render_single(env, policy, max_steps=100, show_rendering=True):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        if show_rendering:
            env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    if show_rendering:
        env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def evaluate(env, policy, max_steps=100, max_episodes=32):
    """
    This function does not need to be modified,
    evaluates your policy over multiple episodes.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_rewards = []
    dones = []
    for _ in range(max_episodes):
        episode_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        episode_rewards.append(episode_reward)
        dones.append(done)

    episode_rewards = np.array(episode_rewards).mean()
    success = np.array(dones).mean()

    print(f"> Average reward over {max_episodes} episodes:\t\t\t {episode_rewards}")
    print(f"> Percentage of episodes goal reached:\t\t\t {success * 100:.0f}%")
