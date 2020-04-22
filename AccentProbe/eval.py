import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_3 import AccentNet
from data_loader import AccentDataset
import argparse
import pdb
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import multiprocessing
import os
from collections import OrderedDict

def custom_collate_fn(batch):
#    batch = torch.tensor(batch)
	# i'd like to pre-move the data to the GPU but i get an error here:
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
def eval(test_csv_path, file_info_path,data_path, rep_type,batch_size, num_epochs, inp_dim0, inp_dim1, all_gpu, model_path):
	cuda = torch.cuda.is_available()
#	cuda1 = torch.device('cuda:1')
	cuda = False
	#train_set = AccentDataset(train_csv_path, file_info_path, data_path, rep_type, all_gpu)
	test_set = AccentDataset(test_csv_path, file_info_path, data_path, rep_type, all_gpu)
	inp_dim = (inp_dim0, inp_dim1)
#	torch.set_num_threads(32)
	net = AccentNet(inp_dim)
	criterion = nn.CrossEntropyLoss()
	if(cuda):
		net = net.cuda()
		criterion = criterion.cuda()
		net = torch.nn.DataParallel(net)
	#if(use_model):
	state_dict = torch.load(model_path)
	#net.load_state_dict(state_dict)
#	pdb.set_trace()
#	if (state_dict.keys()[0].split('.')[0] == 'module'):
#		print(satisfied)
	try:	
		net.load_state_dict(state_dict)
	except:
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]
			new_state_dict[name] = v
		#net = torch.nn.DataParallel(net)
		net.load_state_dict(new_state_dict)
	#print(state_dict)
	#pdb.set_trace()

	if(not all_gpu):
	#	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = multiprocessing.cpu_count()//4, pin_memory = True)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
	else:
	#	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	print('Loading finished')
#	print(torch.get_num_threads())
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
			test_total += label.size(0)
			test_correct += (predicted == label).sum().item()
			#calculate loss
			#calculate accuracy
		print('Test Accuracy: {}'.format(str(100*test_correct/test_total)))
		#torch.save(net.state_dict(), 'Weights_{}/Weights_{}.pth'.format(rep_type, str(epoch+1)))

parser = argparse.ArgumentParser(description='Take command line arguments')

parser.add_argument('--test_main_path',type=str)
parser.add_argument('--file_info_path',type=str)
parser.add_argument('--data_path',type=str)
parser.add_argument('--rep_type',type=str)
#parser.add_argument('--learning_rate',type=float)
parser.add_argument('--batch_size',type=int)
#parser.add_argument('--num_epochs',type=int)
parser.add_argument('--all_gpu', action='store_true', default= False)
#parser.add_argument('--use_model', action='store_true', default = False)
parser.add_argument('--model_path', type= str)
args = parser.parse_args()
dim = {'spec':[161,601], 'conv':[1312,301], 'rnn_0': [1024,301], 'rnn_1': [1024,301], 'rnn_2': [1024, 301], 'rnn_3': [1024, 301], 'rnn_4': [1024,301]}
#pdb.set_trace()
#accent_list = ['us', 'indian', 'african', 'scotland', 'england', 'australia', 'canada']
#for accent in accent_list:
#	test_path = os.path.join(args.test_main_path, '{}_test_accent.csv')
#	eval(test_path, args.file_info_path, args.data_path, args.rep_type, args.batch_size, args.num_epochs, dim[args.rep_type][0], dim[args.rep_type][1],args.all_gpu, args.model_path)
if __name__ == '__main__':
	#eval(args.train_csv_path, args.test_csv_path, args.file_info_path,args.data_path, args.rep_type, args.learning_rate, args.batch_size, args.num_epochs, dim[args.rep_type][0], dim[args.rep_type][1],args.all_gpu, args.use_model, args.model_path)
	accent_list = ['us', 'indian', 'african', 'scotland', 'england', 'australia', 'canada','latest']
	for accent in accent_list:
		test_path = os.path.join(args.test_main_path, '{}_test_accent.csv'.format(accent))
		print(accent)
		eval(test_path, args.file_info_path, args.data_path, args.rep_type, args.batch_size, 1, dim[args.rep_type][0], dim[args.rep_type][1],args.all_gpu, args.model_path)
