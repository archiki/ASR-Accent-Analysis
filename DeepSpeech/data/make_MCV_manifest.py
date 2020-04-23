import argparse
import pdb
import os
import string
import subprocess
from tqdm import tqdm

def make_MCV_manifest(clips_path, target_path, filenames_path):
	txt_path = os.path.join(target_path,'txt')
	wav_path = os.path.join(target_path,'wav')
	if (not os.path.exists(target_path)):
		os.mkdir(target_path)
		os.mkdir(txt_path)
		os.mkdir(wav_path)
	with open(filenames_path, 'r+') as names_file:
		with open("MCV_all.csv", "w+") as m:
			file_names = names_file.readlines()
			for file_name in tqdm(file_names):
#				try:
				[file_name, transcript] = file_name.rstrip("\n").split(' : ')
				transcript = transcript.strip().upper()
				valid_punctuation = string.punctuation.replace("'","")
				transcript = transcript.translate(str.maketrans({a:None for a in valid_punctuation }))
				file_name = file_name.split('.')[0]
				cmd = "sox -q -v 0.98 {} -r {} -b 16 -c 1 {}".format(os.path.join(clips_path,file_name+'.mp3'),16000,os.path.join(wav_path,file_name+'.wav'))
				subprocess.call([cmd], shell=True)
				row = os.path.join(wav_path,file_name+'.wav') +"," + os.path.join(txt_path,file_name+'.txt') + "\n"
				m.write(row)
				with open(os.path.join(txt_path,file_name+'.txt'),'w+') as t:
					t.write(transcript)
#				except:
#					continue





parser = argparse.ArgumentParser(description='makes manifest of MCV-v3')
parser.add_argument('--clips_path', type = str)
parser.add_argument('--target_path', type = str)
parser.add_argument('--filenames_path', type = str)
#parser.add_argument('--manifest_path', type = str)
args = parser.parse_args()
if __name__ == '__main__':
	make_MCV_manifest(args.clips_path, args.target_path, args.filenames_path)
	
