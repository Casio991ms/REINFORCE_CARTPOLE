from torch import nn
import torch.nn.functional as F


class REINFORCE(nn.Module):
    def __init__(self, s_size, h_size, a_size):
        super(REINFORCE, self).__init__()
        self.layer1 = nn.Linear(s_size, h_size)
        # self.layer2 = nn.Linear(h_size, h_size)
        self.layer3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
