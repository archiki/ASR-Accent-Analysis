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
from collections import OrderedDict 

#	print(type(batch[1]))
#	return out_batch
def eval(test_path,rep_type, batch_size, num_epochs, inp_dim0, inp_dim1,model_path, hidden_dim = 500, all_gpu = False):
	cuda = torch.cuda.is_available()
	test_set = PhoneDataset(rep_type, test_path)
	inp_dim = (inp_dim0, inp_dim1)
#	torch.set_num_threads(32)
	net = PhoneNet(inp_dim, hidden_dim)
	criterion = nn.CrossEntropyLoss()
	if(cuda):
		net = net.cuda()
		criterion = criterion.cuda()
		
	
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

		if(not all_gpu):
			test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers = multiprocessing.cpu_count()//4, pin_memory = True)
	else:
		test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	print('Loading finished')
#	
	for epoch in range(num_epochs):
		train_loss = 0
		test_loss = 0
		train_total = 0
		test_total = 0
		train_correct = 0
		test_correct = 0
		net.eval()

		for rep, label in tqdm(test_loader):
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
		print("Test Accuracy {}".format(100*test_correct/test_total))
		#print('Epoch: {}, Train Loss: {}, Test Loss: {}, Train Accuracy: {}, Test Accuracy: {}'.format(str(epoch), str(train_loss/train_total),  \
		#	str(test_loss/test_total), str(100*train_correct/train_total), str(100*test_correct/test_total)))
		#torch.save(net.state_dict(), 'Weights_{}/Weights_{}.pth'.format(rep_type, str(epoch+1)))

parser = argparse.ArgumentParser(description='Take command line arguments')
#parser.add_argument('--train_path',type=str)
parser.add_argument('--test_path',type=str)
parser.add_argument('--rep_type',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--num_epochs',type=int)

parser.add_argument('--use_model', action='store_true', default = False)
parser.add_argument('--model_path', type= str)
args = parser.parse_args()
dim = {'spec':[161,1], 'conv':[1312,1], 'rnn_0': [1024,1], 'rnn_1': [1024,1], 'rnn_2': [1024, 1], 'rnn_3': [1024, 1], 'rnn_4': [1024,1]}
#pdb.set_trace()
if __name__ == '__main__':
	print(args.test_path.split('/')[-2])
	print(args.rep_type)
	eval( args.test_path, args.rep_type, args.batch_size, 1, dim[args.rep_type][0], dim[args.rep_type][1], args.model_path)

