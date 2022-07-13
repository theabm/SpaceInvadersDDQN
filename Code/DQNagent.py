import torch
import pickle
import random

from NeuralNetwork import *
from Memory import *


class DQNAgent():

  def __init__(self, state_space_shape, action_space_size, memory_size, batch_size, eps_max, eps_min, 
               schedule_timesteps, discounting, learning_rate, learning_start, update_method,
               soft_update_param = None, update_freq = None, PATH = None, episode = None):
    assert learning_start >= batch_size, f"{learning_start} must be equal or greater than {batch_size}."
 
    # Environment Parameters
    self.state_space_shape = state_space_shape
    self.action_space_size = action_space_size
    self.discounting = discounting
    self.learning_start = learning_start

    # Experience Replay Parameters
    self.memory_size = memory_size
    self.batch_size = batch_size
    
    # Exploration (epsilon greedy) parameters
    self.eps_max = eps_max
    self.eps_min = eps_min
    self.schedule_timesteps = schedule_timesteps

    # Parameters for set up
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #DQN models. 
    self.policy = DQN(state_space_shape, action_space_size).to(self.device)
    self.target = DQN(state_space_shape, action_space_size).to(self.device)
    
    self.method = update_method
    if self.method == "soft":
      assert soft_update_param is not None, "Specify soft update parameter."
      self.soft_update_param = soft_update_param # Target network will be updated as:  soft_update*Q_model + (1-soft_update)*Q_target
    elif self.method == "hard":
      assert update_freq is not None, "Specify update frequency."
      self.update_freq = update_freq

    # Parameters for neural network
    self.learning_rate = learning_rate
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)
    self.loss = nn.SmoothL1Loss().to(self.device)


    if PATH is not None:
      checkpoint = torch.load(PATH)
      self.policy.load_state_dict(checkpoint['policy_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
      self.target.load_state_dict(checkpoint['target_state_dict'])
      self.step = checkpoint['step']
      self.eps_current = checkpoint['eps_current']
      if episode is not None:
        with open(f"/u/dssc/s271711/fast/Atari/TrainingCheckpoint/memory_{episode}.pt", "rb") as fp:
          self.memory = pickle.load(fp)
      else:
          self.memory = ReplayMemory(capacity = self.memory_size)
    else:
      self.step = 0
      self.memory = ReplayMemory(capacity = self.memory_size)
      self.eps_current = eps_max

  def remember(self, state, action, reward, next_state, done):
    '''Remember an experience'''
    self.memory.push(state,action,reward,next_state,done)

  def epsilon_greedy(self, state):
    '''epsilon greedy strategy to choose best action'''
    self._decrease_exploration()
    if random.random() > self.eps_current:
      with torch.no_grad():
        return int(torch.argmax(self.policy(state.toTorch().unsqueeze(0).to(self.device).to(torch.float))).cpu())
    else:
      return random.randrange(self.action_space_size)
    
  def act(self, state, epsilon):
    if random.random() > epsilon:
      with torch.no_grad():
        return int(torch.argmax(self.policy(state.toTorch().unsqueeze(0).to(self.device).to(torch.float))).cpu())
    else:
      return random.randrange(self.action_space_size)


  def experience_replay(self):
    if len(self.memory) < self.learning_start:
      # Need to acquire more experience
      return
    experiences = self.memory.sample(self.batch_size)

    experiences = Transition(*zip(*experiences))
    #tuple of lazy frames
    next_state_batch = [next_state.toTorch() for next_state in experiences.next_state]
    next_state_batch = torch.stack(next_state_batch).to(self.device)
    reward_batch     = torch.tensor(experiences.reward).to(self.device) # size 32
    done_batch       = torch.tensor(experiences.done, dtype = torch.int).to(self.device)

    with torch.no_grad():
      max_a = torch.argmax(self.policy(next_state_batch.to(torch.float)), axis = 1).unsqueeze(1)
      target = reward_batch + self.discounting * torch.mul(self.target(next_state_batch.to(torch.float)).gather(1, max_a).squeeze(), 1-done_batch)
    
    del next_state_batch
    del reward_batch
    del done_batch
    del max_a
    
    state_batch      = [state.toTorch() for state in experiences.state]
    state_batch      = torch.stack(state_batch).to(self.device)
    
    action_batch     = torch.tensor(experiences.action).unsqueeze(1).to(self.device)
    
    self.optimizer.zero_grad()
    loss = self.loss(target, self.policy(state_batch.to(torch.float)).gather(1, action_batch).squeeze())

    loss.backward()
    self.optimizer.step()

    del state_batch
    del action_batch
    del target
    
    self._update()
    
  def _decrease_exploration(self):
    fraction = min(self.step/self.schedule_timesteps, 1.0)
    
    self.eps_current = (self.eps_min-self.eps_max)*fraction + self.eps_max

    self.step += 1 # Everytime we decrease exploration, we are choosing an action thus taking another step.
    return
  
  def _update(self):
    if self.method == "soft":
      return self._soft_udpate_target()
    elif self.method == "hard":
      return self._hard_update_target()
    else:
      print(f"Update method not recognized. Choose between 'soft' or 'hard'")

  def _soft_udpate_target(self):
   for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
       target_param.data.copy_(self.soft_update_param*param.data + target_param.data*(1.0 - self.soft_update_param))
   return
  
  def _hard_update_target(self):
    if (self.step-self.learning_start)%self.update_freq == 0:
      self.target.load_state_dict(self.policy.state_dict())
    
  def show_status(self, episode, avg_reward):
    print(f"Episode {episode} completed. Average Reward: {avg_reward}")

  def save(self, episode, total_rewards, duration):
    filename = f"/u/dssc/s271711/fast/Atari/TrainingCheckpoint/dqnAgent_{episode}.pt"
    torch.save({
            'episode' : episode,
            'total_rewards': total_rewards,
            'duration': duration,
            'step': self.step,
            'eps_current': self.eps_current,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'target_state_dict': self.target.state_dict()
            }, filename)
    self.memory.save(episode)

