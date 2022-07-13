import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(x.size(0),-1))
        return x