#from torch.utils.data import Dataloader
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
import math
import pdb
import time
import glob
class PhoneDataset(Dataset):
	def __init__(self, rep_type, phones_path):
		
		self.size = len(glob.glob1(phones_path,"*.npy"))
		self.rep_type = rep_type
		self.phones_path = phones_path
		self.list = os.listdir(self.phones_path)
		#self.valid_phones = ['ao', 'ae', 'r', 'eh', 't', 'b', 'aa', 'f', 'k', 'ng', 's', 'g', 'ow', 'er', 'l', 'th', 'z', 'aw', 'd', 'dh', 'sh', 'hh', 'iy', 'ch', 'm', 'ey', 'v', 'y', 'zh', 'jh', 'p', 'uw', 'ah', 'w', 'n', 'oy', 'ay', 'ih', 'uh']
		self.valid_phones = ['aa', 'el', 'ch', 'ae', 'eh', 'ix', 'ah', 'ao', 'w', 'ih', 'tcl', 'en', 'ey', 'ay', 'ax', 'zh', 'er', 'gcl', 'ng', 'nx', 'iy', 'sh', 'pcl', 'uh', 'bcl', 'dcl', 'th', 'dh', 'kcl', 'v', 'hv', 'y', 'hh', 'jh', 'dx', 'em', 'ux', 'axr', 'b', 'd', 'g', 'f', 'k', 'm', 'l', 'n', 'q', 'p', 's', 'r', 't', 'oy', 'ow', 'z', 'uw']


		super(PhoneDataset, self).__init__()

	def __getitem__(self, index):
#		start_time = time.time()
		rep = np.load(os.path.join(self.phones_path, self.list[index]))
		rep = torch.tensor(rep)
		file_name = self.list[index]
		phone_label = file_name.split('_')[-2]
		label = self.valid_phones.index(phone_label)
		label = torch.tensor(label)
		#pdb.set_trace()
		#print(len(self.valid_phones))
		return rep, label 
		

	def __len__(self):
		return self.size



#class AudioDataloader(Dataloader):






