import argparse
import pdb
import numpy as np
import torch
from tqdm import tqdm
import json
import time
from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
import gc
parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
					help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=5, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)

def contrib(layer_op,grad, layer_index, inp):
	inp.grad.data.zero_()
	model.zero_grad()
	#layer_op[:,:,layer_index].backward(grad[:,:,layer_index], retain_graph = True) # for conv
	layer_op[layer_index, :,:].backward(grad[layer_index,:,:], retain_graph = True) # for rnns 
	return inp.grad #check this







def evaluate(test_loader, device, model, decoder, target_decoder,accent, save_output=False, verbose=False, half=False):
	#model.train()
	total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
	output_data = []
	contribution_dict = {}

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
		chars, indices = torch.max(out,dim = 2)
		
		del conv, rnn_3,rnn_4
		gc.collect()
		torch.cuda.empty_cache()
		layers = [rnn_0, rnn_1, rnn_2]
		layer_names = ['rnn_0','rnn_1','rnn_2']

		indices_list = torch.unbind(indices, dim = 0)
		batch_final = torch.zeros(chars.shape[0])
		for i in range(inputs.shape[0]):
			
			final_chars = chars[i]
			final = torch.sum(final_chars)
			batch_final[i] = final

		batch_final = torch.sum(batch_final)
#		start = time.process_time()	
		batch_final.backward(retain_graph = True)
#		print('done')
#		print(time.process_time() - start)
		l = 0
		for layer in layers:
			#print(layer_names[l])
			grad_copy = layer.grad.detach().clone()
			files_dict = {}
			
			for j in range(output_sizes[0]):
#					
				model.zero_grad()
				contribution = contrib(layer,grad_copy, j, inputs)
				contribution = contribution.view(inputs.shape[0],-1, inputs.shape[-1])
				contribution = torch.norm(contribution, dim = 1)
#				
				for i in  range(inputs.shape[0]):
					if(j >= output_sizes[i]):
						continue
					c = contribution[i]
					denom = torch.sum(c)
					if(torch.sum(c) == 0):
						denom = 1e-09
					contr = c/denom
					np.save('../data/Contribution/timit_blank/{}/{}_{}_{}'.format(layer_names[l],filenames[i], str(j),layer_names[l]), contr.cpu().numpy())
					
					del c
					del contr
#					
					gc.collect()
					torch.cuda.empty_cache()
				del contribution
				gc.collect()
				torch.cuda.empty_cache()

			l += 1
			del layer
			del grad_copy
			gc.collect()
			torch.cuda.empty_cache()
			
		if save_output:
			# add output to data array, and continue
			output_data.append((out.cpu().numpy(), output_sizes.numpy()))


	return

if __name__ == '__main__':
	args = parser.parse_args()
	torch.set_grad_enabled(True)
	device = torch.device("cuda" if args.cuda else "cpu")
	accent = args.test_manifest.split('/')[-1].split('_')[1]

	model = load_model(device, args.model_path, args.half)

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
								  num_workers=args.num_workers)
	wer, cer, output_data = evaluate(test_loader=test_loader,
									 device=device,
									 model=model,
									 decoder=decoder,
									 target_decoder=target_decoder,
									accent = accent,
									 save_output=args.save_output,
									 verbose=args.verbose,
									 half=args.half)

	print('Test Summary \t'
		  'Average WER {wer:.3f}\t'
		  'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
	if args.save_output is not None:
		np.save(args.save_output, output_data)
