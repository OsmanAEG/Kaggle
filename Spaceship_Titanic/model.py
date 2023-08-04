import torch
from torch import nn
import torch.nn.functional as F

class Model_Arch(nn.Module):
  def __init__(self, N_in, N_out):
    super(Model_Arch, self).__init__()

    self.l_in  = nn.Linear(N_in, 60)
    self.l_h1  = nn.Linear(60, 60)
    self.l_h2  = nn.Linear(60, 60)
    self.l_h3  = nn.Linear(60, 60)
    self.l_out = nn.Linear(60, N_out)

  def forward(self, x):
    x = F.relu(self.l_in(x))
    x = F.relu(self.l_h1(x))
    x = F.relu(self.l_h2(x))
    x = F.relu(self.l_h3(x))
    x = torch.sigmoid(self.l_out(x))

    return x




