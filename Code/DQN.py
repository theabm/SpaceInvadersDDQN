import numpy as np           
from skimage import transform 
from collections import deque, namedtuple
import random
import pickle

import gym

import torch


from Memory import *
from NeuralNetwork import *
from EnvWrapper import *
from DQNagent import *

env = gym.make(id='SpaceInvadersNoFrameskip-v4')
env = FrameStack(ClipRewardEnv(MaxAndSkipEnv(ProcessObservation(env))))


def trainAgent(episodes = 100000, skipsteps = 1):
  agent = DQNAgent(
      state_space_shape = [4,84,84],
      action_space_size = 6,
      memory_size = 100000,
      batch_size = skipsteps*32,
      eps_max = 1.0,
      eps_min = 0.05,
      schedule_timesteps = 100000,
      discounting = 0.99,
      learning_rate = 5e-4,
      learning_start = 5000,
      update_method="soft",
      soft_update_param = 0.002
      )
  total_rewards = []
  duration = []
  for episode in range(1, episodes):
    state = env.reset()
    total_reward = np.array(0).astype(np.int16)
    steps = 0
    skipsteps = skipsteps
    while True:
      
      #sample action using epsilon greedy
      action = agent.epsilon_greedy(state)

      #take action and receive reward and next state
      next_state, reward, done, _ = env.step(action)
      total_reward += reward
      steps += 1
      skipsteps -= 1
      
      #store experience
      agent.remember(state,action,reward,next_state,done)

      #Experience Replay to update Policy model
      if skipsteps==0:
        agent.experience_replay()
        skipsteps = 4
      
      state = next_state
      if done:
        break
      
    total_rewards.append(total_reward)
    duration.append(steps)

    if episode%500 == 0:
      agent.save(episode, total_rewards, duration)


trainAgent(skipsteps = 4)