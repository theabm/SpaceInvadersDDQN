from collections import deque, namedtuple
import random
import pickle


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():

    def __init__(self, capacity = 1000000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size = 32):
        return random.sample(self.memory, batch_size)
    
    def save(self, episode):
      filename = f"/u/dssc/s271711/fast/Atari/TrainingCheckpoint/memory_{episode}.pt"
      with open(filename, "wb") as fp:
        pickle.dump(self, fp)

    def __len__(self):
        return len(self.memory)
    
    def __iter__(self):
      return ReplayMemoryIterator(self)

class ReplayMemoryIterator():
  def __init__(self, replay_memory):
    self.replaymemory = replay_memory.memory
    self.index = 0
  def __iter__(self):
    return self
  def __next__(self):
    if self.index < len(self.replaymemory):
      transition = self.replaymemory[self.index]
      self.index = self.index + 1
      return transition
    else:
      raise StopIteration
