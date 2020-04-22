import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import pdb
import torch.nn.functional as F

class AccentNet(nn.Module):
	def __init__(self, input_dim):
		super(AccentNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1,1, kernel_size=(31,21), stride=(3, 2), padding=(15,10)), #prev 21,11 stride = 2,2
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.MaxPool2d((5,3), stride=(3, 2) )) #prev 2,2 , 1,1
		self.layer2 = nn.Sequential(
			nn.Conv2d(1,1,kernel_size=(11,5), stride=(2,1), padding = (5,2)), #prev 11,5 stride = 2,1
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.MaxPool2d((3,2), stride=(2,1))) #prev 3,2 , 2,1
		required_linear1 = int(math.floor((input_dim[0] + 2 * 15 -31)/3) + 1)
		required_linear1 = int(math.floor((required_linear1 - 5)/3) + 1)
		required_linear1 = int(math.floor((required_linear1 - 11 +5*2)/2) + 1)
		required_linear1 = int(math.floor((required_linear1 -3)/2) +1)
		required_linear2 = int(math.floor((input_dim[1] + 2*10 - 21 )/2) + 1)
		required_linear2 = int(math.floor((required_linear2 -3 )/2) + 1)
		required_linear2 = int(math.floor((required_linear2 + 2*2 -5 )/1) + 1)
		required_linear2 = int(math.floor((required_linear2 -2 )/1) + 1)
		self.drop_out1 = nn.Dropout(p=0.4)
		self.drop_out2 = nn.Dropout(p=0.4)
		self.drop_out3 = nn.Dropout(p=0.4)
		self.fc1 = nn.Linear(required_linear1*required_linear2, 500)
		self.fc = nn.Linear(500, 7)

	def forward(self, x):
#		pdb.set_trace()
		x = x.unsqueeze(1)
		out = self.layer1(x)
		out = self.drop_out3(out)
		out = self.layer2(out)
#		pdb.set_trace()
		out = out.view(out.size(0), -1)
		out = self.drop_out1(out)
		out =F.relu(self.fc1(out))
		out = self.drop_out2(out)
		out = self.fc(out)
		return out
