import argparse
import pdb
import numpy as np
import torch
from tqdm import tqdm
import json
from itertools import groupby
from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
import time
import pickle
import torch.distributed as dist
import torch.nn as nn
import gc
import os
from apex.parallel import DistributedDataParallel
parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
					help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=5, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser = add_decoder_args(parser)

def contrib(layer_op,grad,batch_idx, layer_index, inp):
	inp.grad.data.zero_()
	model.zero_grad()
	layer_op[batch_idx, :,layer_index].backward(grad[batch_idx,:,layer_index], retain_graph = True)
	return inp.grad 







def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
	dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


	model.eval()

	for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

		inputs, targets, input_percentages, target_sizes, filenames = data # see if convert to variable and set requires_grad to true
		input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
		inputs = inputs.to(device)
		if half:
			inputs = inputs.half()
		# unflatten targets
		split_targets = []
		offset = 0
		for size in target_sizes:
			split_targets.append(targets[offset:offset + size])
			offset += size
	
		inputs.requires_grad = True
		torch.set_grad_enabled(True)
		out, output_sizes, conv, rnn_0, rnn_1, rnn_2, rnn_3, rnn_4 = model(inputs, input_sizes)
		
		del input_percentages
		del conv
		del rnn_0
		del rnn_1 
		del rnn_2
		del rnn_3
		del rnn_4
		gc.collect()
		torch.cuda.empty_cache()
		chars, indices = torch.max(out,dim = 2)
		indices = indices.detach().cpu().numpy()
		target_path = '/workspace/data/attribution/timit/' 	

		for i in range(inputs.shape[0]): # batchsize
			if( os.path.exists(target_path + filenames[i] + '.pickle')):
				continue
			
			char_dict = {}
			grad_char_dict = {}
			
			group_list = [list(group) for k, group in groupby(indices[i][:output_sizes[i]])]
			combine_list = [0]
			idx_list = []
			for g in group_list:
				combine_list.append(combine_list[-1]+ len(g))
				idx_list.append(g[0])
			final = []
			del group_list
			for j in range(len(idx_list)):
				final.append(torch.sum(chars[i][combine_list[j]:combine_list[j+1]]))
			final_char = torch.stack(final)
			del combine_list		
			for k in range(len(idx_list)):
				
				if(idx_list[k] == 0):
					continue
				if(k!= len(idx_list)):
					final_char[k].backward(retain_graph = True)
				else:
					final_char[k].backward()
				inp_grad = inputs.grad[i].detach().clone().view(-1,inputs.shape[-1])
				#print(inp_grad.shape)
				model.zero_grad()
				inputs.grad.data.zero_()
				attr_grad = torch.norm(inp_grad, dim = 0)
				attr = torch.mul(inputs[i].view(-1,inputs.shape[-1]), inp_grad)
				attr = torch.sum(attr, dim = 0)
				attr_cpu = attr.detach().cpu().numpy()
				attr_grad_cpu = attr_grad.detach().cpu().numpy()
				char_dict[k] = attr_cpu
				grad_char_dict[k] = attr_grad_cpu
				del attr
				del inp_grad
				del attr_cpu
				del attr_grad_cpu
				gc.collect()
				torch.cuda.empty_cache()
				
			str_l = ["_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "[m] for m in idx_list]
			str_op = ''.join(str_l)
			
			my_dict = {'output':str_op, 'attr dict':char_dict, 'grad_dict':grad_char_dict}
	#		print(time.process_time() - start)
			del str_l
			del str_op
			with open('../data/attribution/timit/{}.pickle'.format(filenames[i]),'wb+') as file:
				pickle.dump(my_dict, file)
			del final_char
			del final
			del idx_list
			del my_dict
			del char_dict
			del grad_char_dict
			gc.collect()
			torch.cuda.empty_cache()

		del data
		del inputs 
		del targets
		del input_sizes
		del out 
		del output_sizes
#		del target
#		del target_sizes
		del chars	
		gc.collect()	
		torch.cuda.empty_cache()
	wer = float(total_wer) / num_tokens
	cer = float(total_cer) / num_chars
	
	

	return wer * 100, cer * 100, output_data


if __name__ == '__main__':
	args = parser.parse_args()
	torch.set_grad_enabled(True)
	device = torch.device("cuda" if args.cuda else "cpu")

	model = load_model(device, args.model_path, args.half)
	model.to(device)
	if args.decoder == "beam":
		from decoder import BeamCTCDecoder

		decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
								 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
								 beam_width=args.beam_width, num_processes=args.lm_workers)
	elif args.decoder == "greedy":
		decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
	else:
		decoder = None
	target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
	test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
									  labels=model.labels, normalize=True)
	test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
								  num_workers=args.num_workers, shuffle= False)
	wer, cer, output_data = evaluate(test_loader=test_loader,
									 device=device,
									 model=model,
									 decoder=decoder,
									 target_decoder=target_decoder,
								#	accent = accent,
									 save_output=args.save_output,
									 verbose=args.verbose,
									 half=args.half)

	print('Test Summary \t'
		  'Average WER {wer:.3f}\t'
		  'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
	if args.save_output is not None:
		np.save(args.save_output, output_data)
