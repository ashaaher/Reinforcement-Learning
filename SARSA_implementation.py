#!/usr/bin/env python
# coding: utf-8

# # SARSA algorithm implementation for mountain car problem using Gym

# SARSA algorithm implementation for mountain car problem using Gym
# Understanding on SARSA: The State–Action–Reward–State–Action (SARSA) algorithm is an on-policy learning problem. Just like Q-learning, SARSA is also a temporal difference learning problem, meaning, it looks ahead at the next step in the episode to estimate future rewards. The major difference between SARSA and Q-learning is that the action having the maximum Q-value is not used to update the Q-value of the current state-action pair. Instead, the Q-value of the action as the result of the current policy, or owing to the exploration step like -greedy is chosen to update the Q-value of the current state-action pair The name SARSA comes from the fact that the Q-value update is done by using a quintuple Q(s,a,r,s',a') where:
# 
# 
# 
# s,a: current state and action
# 
# r: reward observed post taking action a
# 
# s': next state reached after taking action a
# 
# a': action to be performed at state s'
# 
# 

# **Importing the dependencies and examine the mountain car environment**

# In[31]:


import gym
import numpy as np


# **Assigning the hyperparameters**
# 
# such as number of states, number of episodes, learning rate (both initial and minimum), discount factor gamma, maximum steps in an episode, and the epsilon for epsilon-greedy.

# In[ ]:


#!/usr/bin/env/ python
"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. |Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
#MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 6000
STEPS_PER_EPISODE = 200 #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim


class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))  # (51 x 51 x 3)
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            discretized_obs = self.discretize(obs)
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])
        
#Till here every task has been similar to the Q-learning algorithm. Now the SARSA implementation starts with initializing the Q-table and updating the Q-values accordingly, 
#In below code we have updated the reward value as an absolute difference between the current position and position at the lowest point, that is, start point so that it maximizes the reward by going away from the central, that is, lowest point.
   
    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        next_action = agent.get_action(next_obs)
        td_target = reward + self.gamma * self.Q[discretized_next_obs][next_action]
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error
        return next_action

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        action = agent.get_action(obs)
        while not done:
            next_obs, reward, done, info = env.step(action)
            next_action = agent.learn(obs, action, reward, next_obs)
            action = next_action
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))
    # Return the trained policy
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()

