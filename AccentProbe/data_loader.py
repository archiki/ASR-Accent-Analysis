#from torch.utils.data import Dataloader
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
import math
import pdb
import time
class AccentDataset(Dataset):
	def __init__(self, csv_path, file_info_path, data_path, rep_type,all_gpu):
		with open(csv_path, 'r') as f:
			ids = f.readlines()
		ids = [x.strip().split(',') for x in ids]
		self.ids = ids
		self.size = len(ids)
		self.rep_type = rep_type
		self.rep_path = os.path.join(data_path, rep_type)
		#self.file_info_path = file_info_path
		with open(file_info_path, 'r') as j:
			file_meta = json.load(j)
		self.file_meta = file_meta
		self.all_gpu = all_gpu

		super(AccentDataset, self).__init__()

	def __getitem__(self, index):
#		start_time = time.time()
		sample = self.ids[index]
		file_name, accent_label, duration = sample[0], sample[1], sample[2]
		representation = np.load(os.path.join(self.rep_path, file_name + '_{}.npy'.format(self.rep_type)))
		representation = torch.tensor(representation)
		times = self.file_meta[file_name]['end_times']
		start_sil_dur = times[0]
		end_sil_dur = times[-1] - times[-2]
#		pdb.set_trace()
		
		silence_removed_representaion = self.process_silence(representation, start_sil_dur, end_sil_dur)
		accent_list = ['us', 'indian', 'england', 'scotland', 'australia', 'african', 'canada']
		accent_label_arr = np.zeros(len(accent_list))
		idx = accent_list.index(accent_label)
		idx = torch.tensor(idx)
		#print("--- %s Item Seconds ---" % (time.time() - start_time))
		if(self.all_gpu):
			return silence_removed_representaion.to('cuda', non_blocking = True), idx.to('cuda', non_blocking = True)
		else:
			return silence_removed_representaion, idx

	def process_silence(self,representation, start_sil_dur, end_sil_dur):
		Fs = 16000
		window_stride = 0.01
		max_durr = 6
		starting_frames =int( math.ceil((start_sil_dur*Fs)/(window_stride*Fs)))
		ending_frames = int(math.ceil((end_sil_dur*Fs)/(window_stride*Fs)))
		max_frames = int(math.ceil((max_durr*Fs)/(window_stride*Fs)) + 1)
		if(self.rep_type != "spec"):
			starting_frames = int(math.floor((starting_frames -11 + 2*5)/2) + 1)
			ending_frames = int(math.floor((ending_frames -11 + 2*5)/2) + 1)
			max_frames = int(math.floor((max_frames -11 + 2*5)/2) + 1)

		feat_dim , rep_size = representation.shape
		sil_removed_rep = representation[:,starting_frames: rep_size - ending_frames]
		feat_dim , rem_rep_size = sil_removed_rep.shape

		if(rem_rep_size >= max_frames):
			final_rep = sil_removed_rep[:,:max_frames]
		else:
			final_rep = torch.zeros(feat_dim, max_frames)
			final_rep[:,:rem_rep_size] = sil_removed_rep
		return final_rep


	def __len__(self):
		return self.size



#class AudioDataloader(Dataloader):






