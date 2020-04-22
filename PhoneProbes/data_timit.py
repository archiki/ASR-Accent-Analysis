import numpy as np
import torch
import os
import json
import pickle
import math
import pdb
import time
import pdb
import argparse
from tqdm import tqdm

def get_input_frame(current_frame):
	return (current_frame - 1)*2 + 11 - 2*5

def data_prepare(csv_path, file_info_path, data_path, rep_type, target_path):
	with open(csv_path, 'r') as f:
		ids = f.readlines()
		ids = [x.strip().split(',') for x in ids]
		#self.ids = ids
		samp_rate = 16000
		spec_stride = 0.01
		window_size = 0.02
		size = len(ids)
		rep_path = os.path.join(data_path, rep_type)
		#self.file_info_path = file_info_path
		with open(file_info_path, 'rb') as j:
			file_meta = pickle.load(j)


		for i in tqdm(range(size)):
			sample = ids[i]
			file_name, accent_label = sample[0], sample[1]
			#accent_label = 'test'
			file_name = file_name.split('/')[-1].split('.')[0]
			representation = np.load(os.path.join(rep_path, file_name + '_{}.npy'.format(rep_type)))
			representation = torch.from_numpy(representation)
			times = file_meta[file_name]['times']
			rep_list = torch.unbind(representation, dim=1)
			accent_path = target_path
			if not os.path.exists(accent_path):
				os.makedirs(accent_path)
			
			valid_phone_list = ['aa', 'el', 'ch', 'ae', 'eh', 'ix', 'ah', 'ao', 'w', 'ih', 'tcl', 'en', 'ey', 'ay', 'ax', 'zh', 'er', 'gcl', 'ng', 'nx', 'iy', 'sh', 'pcl', 'uh', 'bcl', 'dcl', 'th', 'dh', 'kcl', 'v', 'hv', 'y', 'hh', 'jh', 'dx', 'em', 'ux', 'axr', 'b', 'd', 'g', 'f', 'k', 'm', 'l', 'n', 'q', 'p', 's', 'r', 't', 'oy', 'ow', 'z', 'uw']
			count_dict = dict([(key, 0) for key in valid_phone_list])
			count = 0 
			#samp
			for i in range(len(rep_list)):
				frame_idx = i
				if(rep_type != 'spec'):
					frame_idx = get_input_frame(frame_idx)
				window_start = frame_idx*spec_stride*samp_rate
				
				window_mid = window_start + (samp_rate*window_size/2)
				#print(window_start, window_mid)
				alligned_phone = 'na'
				for j in range(len(times)):
					#print(window_mid, times[j])
					if (window_mid < float(times[j])):
						alligned_phone = file_meta[file_name]['phones'][j]
						
						break
				#print(alligned_phone)
				if(alligned_phone == 'na'):
					#print ("Oops error in allignment for ", file_name, "frame ",frame_idx )
					pass
				if(alligned_phone in valid_phone_list):
					count_dict[alligned_phone] += 1
					

					path = os.path.join(accent_path, file_name+'_'+rep_type+'_'+alligned_phone+'_'+str(count_dict[alligned_phone]))
					#print(alligned_phone)
					np.save(path, rep_list[i].numpy())
			

		return
parser = argparse.ArgumentParser(description='Take command line arguments')
parser.add_argument('--csv_path',type=str)
parser.add_argument('--file_info_path',type=str)
parser.add_argument('--data_path',type=str)
parser.add_argument('--rep_type',type=str)
parser.add_argument('--target_path',type=str)
args = parser.parse_args()

if __name__ == '__main__':
	data_prepare(args.csv_path, args.file_info_path, args.data_path, args.rep_type, args.target_path)







