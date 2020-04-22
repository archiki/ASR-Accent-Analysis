import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import PhoneNet
from data_loader import PhoneDataset
import argparse
import pdb
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import multiprocessing
import os
import json

def custom_collate_fn(batch):
	batch_size = len(batch)
	out_batch = []
	for x in range(batch_size):
		sample = batch[x]
		data = sample[0].to('cuda', non_blocking = True)
		label = torch.tensor(sample[1])
		label = label.to('cuda', non_blocking = True)
		out_batch.append((data,label))

#	print(type(batch[1]))
	return out_batch
def eval_phones(test_path,rep_type, batch_size, num_epochs, inp_dim0, inp_dim1, model_path, hidden_dim = 500, all_gpu=False):
	cuda = torch.cuda.is_available()

	test_set = PhoneDataset(rep_type, test_path)
	inp_dim = (inp_dim0, inp_dim1)
#	torch.set_num_threads(32)
	net = PhoneNet(inp_dim, hidden_dim)
	criterion = nn.CrossEntropyLoss()
	if(cuda):
		net = net.cuda()
		criterion = criterion.cuda()
		net = torch.nn.DataParallel(net)
	state_dict = torch.load(model_path)
	try:
		net.load_state_dict(state_dict)
	except:
		new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:]
				new_state_dict[name] = v
			   #net = torch.nn.DataParallel(net)
				net.load_state_dict(new_state_dict)


	valid_phones = ['ao', 'ae', 'r', 'eh', 't', 'b', 'aa', 'f', 'k', 'ng', 's', 'g', 'ow', 'er', 'l', 'th', 'z', 'aw', 'd', 'dh', 'sh', 'hh', 'iy', 'ch', 'm', 'ey', 'v', 'y', 'zh', 'jh', 'p', 'uw', 'ah', 'w', 'n', 'oy', 'ay', 'ih', 'uh']
	phone_dict = {x:0 for x in valid_phones}
	phone_accuracy = {x:0 for x in valid_phones}
	confusion_dict = {x:phone_dict for x in valid_phones}


	if(not all_gpu):
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = multiprocessing.cpu_count()//4, pin_memory = True)
	else:
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	print('Loading finished')
	for epoch in range(num_epochs):
		
		train_loss = 0
		test_loss = 0
		train_total = 0
		test_total = 0
		train_correct = 0
		test_correct = 0
#		
		net.eval()
#		print("----------- %s Pass seconds --------------" % (time.time() - start_time))
		for rep, label in test_loader:
			rep = Variable(rep)
			label = Variable(label)
			if(cuda):
				rep = rep.cuda()
				label = label.cuda()

			pred = net(rep)
			tloss = criterion(pred, label)
			test_loss += tloss.item()
			_, predicted = torch.max(pred.data, 1)
			gt = valid_phones[label]
			y = valid_phones[predicted]
			confusion_dict[gt][y] += 1
			#test_total += label.size(0)
			#test_correct += (predicted == label).sum().item()
			#calculate loss
			#calculate accuracy
	accent = test_path.split('/')[-2]
	with open('confusion_{}_{}.json'.format(rep_type, accent), 'w+') as f:
		json.dump(confusion_dict, f)
		
parser = argparse.ArgumentParser(description='Take command line arguments')
parser.add_argument('--test_path',type=str)
parser.add_argument('--rep_type',type=str)
parser.add_argument('--batch_size',type=int)

parser.add_argument('--model_path', type= str)
args = parser.parse_args()
dim = {'spec':[161,1], 'conv':[1312,1], 'rnn_0': [1024,1], 'rnn_1': [1024,1], 'rnn_2': [1024, 1], 'rnn_3': [1024, 1], 'rnn_4': [1024,1]}

if __name__ == '__main__':
	torch.manual_seed(0)
	eval_phones(args.test_path, args.rep_type, args.batch_size, 1, dim[args.rep_type][0], dim[args.rep_type][1],args.model_path)

