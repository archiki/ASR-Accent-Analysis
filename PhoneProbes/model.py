import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import pdb
import torch.nn.functional as F

class PhoneNet(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(PhoneNet, self).__init__()
		self.fc1 = nn.Linear(input_dim[0]*input_dim[1], 500)
		self.fc = nn.Linear(hidden_dim, 55)#previous 39
		self.drop_out = nn.Dropout(p = 0.4)

	def forward(self, x):

		out =F.relu(self.fc1(x))
		out = self.drop_out(out)
		out = self.fc(out)
		return out

