import gym
from skimage import transform 
import numpy as np           
import random
from collections import deque, namedtuple
import torch


class ProcessObservation(gym.ObservationWrapper):
  '''Pre process observation by converting frame from RGB to grayscale and then resizing it to (84 x 84)
  '''
  def __init__(self, env):
    super(ProcessObservation, self).__init__(env)
    self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (84,84))
    
  def observation(self, color_frame):
    return ProcessObservation.rgb2grayScaled(color_frame)
  
  @staticmethod
  def rgb2grayScaled(color_frame):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray_frame = np.dot(color_frame[...,:3], rgb_weights)
    cropped_frame = gray_frame[25:-12,4:-12]
    cropped_frame = transform.resize(cropped_frame, [84,84])
    return cropped_frame.astype(np.uint8)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward).astype(np.int16)


# Create our environment wrapper to properly skip the frames.
class MaxAndSkipEnv(gym.Wrapper):

  """
    Each action of the agent is repeated over skip frames        
    return only every `skip`-th frame
  """
  
  def __init__(self, env, skip=3):
    super(MaxAndSkipEnv, self).__init__(env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = deque(maxlen=2)
    self._skip = skip

  def step(self, action):
    total_reward = 0
    for _ in range(self._skip):
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)
        total_reward += reward
        if done:
            break
    max_frame = np.max(np.stack(self._obs_buffer), axis=0)
    return max_frame, total_reward, done, info

  def reset(self):
    """Clear past frame buffer and init to first obs"""
    self._obs_buffer.clear()
    obs = self.env.reset()
    self._obs_buffer.append(obs)
    return obs

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def toTorch(self):
      return torch.from_numpy(np.stack(self._frames))        

    def __len__(self):
      return len(self.toTorch())

    def __getitem__(self, i):
        return self.toTorch()[i]
    
    def shape(self):
      return self.toTorch().shape
  
class FrameStack(gym.Wrapper):
  def __init__(self, env, size = 4):
      super(FrameStack, self).__init__(env)
      self.size = size
      self.queue = deque([], maxlen = self.size)
      shp = env.observation_space.shape
      self.observation_space = gym.spaces.Box(low=0, high=255, shape=((size,) + shp), dtype=env.observation_space.dtype)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.size):
      self.queue.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.queue.append(ob)
    return self._get_ob(), reward, done, info
  
  def _get_ob(self):
    #return torch.from_numpy(np.stack(self.queue, axis = 0))
    return LazyFrames(list(self.queue))

if __name__ == "__main__":
  env = gym.make(id='SpaceInvadersNoFrameskip-v4')
  env = FrameStack(ClipRewardEnv(MaxAndSkipEnv(ProcessObservation(env))))
  
  print("State size:\t", env.observation_space.shape) #The frame size corresponds to the state space. 
  print("Actions:\t ", env.action_space.n)
  
  print("Type of state:\t\t\t\t", type(env.reset()))
  print("Shape of state as torch tensor:\t\t", env.reset().toTorch().shape)
  print("Data type contained in torch tensor:\t", env.reset().toTorch().dtype)
  
  def random_play(close_environment = True):
    score = np.array(0).astype(np.int16)
    env.reset()
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    if close_environment:
      env.close()
      
  random_play()