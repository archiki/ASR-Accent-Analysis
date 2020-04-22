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
def train(train_path, test_path,rep_type,learning_rate, batch_size, num_epochs, inp_dim0, inp_dim1,use_model, model_path, hidden_dim = 500, all_gpu=False):
	cuda = torch.cuda.is_available()

	train_set = PhoneDataset( rep_type, train_path)
	test_set = PhoneDataset(rep_type, test_path)
	inp_dim = (inp_dim0, inp_dim1)
#	torch.set_num_threads(32)
	net = PhoneNet(inp_dim, hidden_dim)
	criterion = nn.CrossEntropyLoss()
	if(cuda):
		net = net.cuda()
		criterion = criterion.cuda()
		#net = torch.nn.DataParallel(net)
	if(use_model):
		state_dict = torch.load(model_path)
		net.load_state_dict(state_dict)
#	torch.distributed.init_process_group(backend="nccl")
#	net = torch.nn.DataParallel(net)
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

	if(not all_gpu):
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = multiprocessing.cpu_count()//4, pin_memory = True)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = multiprocessing.cpu_count()//4, pin_memory = True)
	else:
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	print('Loading finished')

	for epoch in range(num_epochs):
		train_loss = 0
		test_loss = 0
		train_total = 0
		test_total = 0
		train_correct = 0
		test_correct = 0

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
			pred = net(rep)
			loss = criterion(pred, label)
			train_loss += loss.item()
			_, predicted = torch.max(pred.data, 1)
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
		print('Epoch: {}, Train Loss: {}, Test Loss: {}, Train Accuracy: {}, Test Accuracy: {}'.format(str(epoch), str(train_loss/train_total),  \
			str(test_loss/test_total), str(100*train_correct/train_total), str(100*test_correct/test_total)))
		torch.save(net.state_dict(), 'Timit_weights/Iter_2/Weights_{}/Weights_{}.pth'.format(rep_type, str(epoch+1)))

parser = argparse.ArgumentParser(description='Take command line arguments')
parser.add_argument('--train_path',type=str)
parser.add_argument('--test_path',type=str)
parser.add_argument('--rep_type',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--num_epochs',type=int)
#parser.add_arfument('--all_gpu', type=)

parser.add_argument('--use_model', action='store_true', default = False)
parser.add_argument('--model_path', type= str)
args = parser.parse_args()
dim = {'spec':[161,1], 'conv':[1312,1], 'rnn_0': [1024,1], 'rnn_1': [1024,1], 'rnn_2': [1024, 1], 'rnn_3': [1024, 1], 'rnn_4': [1024,1]}

if __name__ == '__main__':
	train(args.train_path, args.test_path, args.rep_type, args.learning_rate, args.batch_size, args.num_epochs, dim[args.rep_type][0], dim[args.rep_type][1], args.use_model, args.model_path)

