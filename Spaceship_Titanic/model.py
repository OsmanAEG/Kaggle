import torch
from torch import nn
import torch.nn.functional as F

class Model_Arch(nn.Module):
  def __init__(self, N_in, N_out):
    super(Model_Arch, self).__init__()

    self.l_in  = nn.Linear(N_in, 128)
    self.l_h1  = nn.Linear(128, 32)
    self.l_out = nn.Linear(32, N_out)

  def forward(self, x):
    x = F.relu(self.l_in(x))
    x = F.relu(self.l_h1(x))
    x = torch.sigmoid(self.l_out(x))

    return x




