import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import AccentNet
from data_loader import AccentDataset
import argparse
import os
import pdb
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import multiprocessing
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
def train(train_csv_path, test_csv_path, file_info_path,data_path, rep_type,learning_rate, batch_size, num_epochs, inp_dim0, inp_dim1, all_gpu, use_model, model_path, iter_num):
	cuda = torch.cuda.is_available()
#	cuda1 = torch.device('cuda:1')
#	cuda = False
	train_set = AccentDataset(train_csv_path, file_info_path, data_path, rep_type, all_gpu)
	test_set = AccentDataset(test_csv_path, file_info_path, data_path, rep_type, all_gpu)
	inp_dim = (inp_dim0, inp_dim1)
#	torch.set_num_threads(32)
	net = AccentNet(inp_dim)
	criterion = nn.CrossEntropyLoss()
	if(cuda):
		net = net.cuda()
		criterion = criterion.cuda()
		net = torch.nn.DataParallel(net)
	if(use_model):
		state_dict = torch.load(model_path)
		net.load_state_dict(state_dict)
#	torch.distributed.init_process_group(backend="nccl")
#	net = torch.nn.DataParallel(net)
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	if(not all_gpu):
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
	else:
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
#		optimizer.zero_grad()
		net.train()
#		start_time = time.time()
		for rep, label in tqdm(train_loader):
#			print(torch.get_num_threads())
#			print('iter')
#			print("--- %s seconds ----" % (time.time() - start_time))
#			start_time = time.time()
			optimizer.zero_grad()
			rep = Variable(rep)
			label = Variable(label)
			if(cuda):
				rep = rep.cuda()
				label = label.cuda()
#			pdb.set_trace()
			pred = net(rep)
			loss = criterion(pred, label)
			train_loss += loss.item()
			_, predicted = torch.max(pred.data, 1)
			#pdb.set_trace()
#			print(predicted)
			train_total += label.size(0)
			train_correct += (predicted == label).sum().item()
			#calculate loss
			#calculate accuracy
			loss.backward()
			optimizer.step()
		
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
		if not os.path.exists('Weights_{}/Iter_{}'.format(rep_type, str(iter_num))):
			os.makedirs('Weights_{}/Iter_{}'.format(rep_type, str(iter_num)))
		print('Epoch: {}, Train Loss: {}, Test Loss: {}, Train Accuracy: {}, Test Accuracy: {}'.format(str(epoch), str(train_loss/train_total),  \
			str(test_loss/test_total), str(100*train_correct/train_total), str(100*test_correct/test_total)))
		torch.save(net.state_dict(), 'Weights_{}/Iter_{}/Weights_{}.pth'.format(rep_type,str(iter_num), str(epoch+1)))

parser = argparse.ArgumentParser(description='Take command line arguments')
parser.add_argument('--train_csv_path',type=str)
parser.add_argument('--test_csv_path',type=str)
parser.add_argument('--file_info_path',type=str)
parser.add_argument('--data_path',type=str)
parser.add_argument('--rep_type',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--all_gpu', action='store_true', default= False)
parser.add_argument('--use_model', action='store_true', default = False)
parser.add_argument('--model_path', type= str)
parser.add_argument('--iter_num', type=str)
args = parser.parse_args()
dim = {'spec':[161,601], 'conv':[1312,301], 'rnn_0': [1024,301], 'rnn_1': [1024,301], 'rnn_2': [1024, 301], 'rnn_3': [1024, 301], 'rnn_4': [1024,301]}
#pdb.set_trace()
if __name__ == '__main__':
	train(args.train_csv_path, args.test_csv_path, args.file_info_path,args.data_path, args.rep_type, args.learning_rate, args.batch_size, args.num_epochs, dim[args.rep_type][0], dim[args.rep_type][1],args.all_gpu, args.use_model, args.model_path, args.iter_num)
