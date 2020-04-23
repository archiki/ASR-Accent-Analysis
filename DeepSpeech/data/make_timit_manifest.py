import os 
import argparse
import string
import librosa

target_dir = 'timit/train/'# change
file_record = {}
wav_file = []
txt_file = []
allign_file = []
m = open('TIMIT_train.csv','w+')
for root, directory, files in os.walk(target_dir):
	#file_path.append(os.path.join(root,filer fo)
	temp = [root+'/'+file for file in files]
	if not os.path.exists('timit_train'):
		os.makedirs('timit_train')
		os.makedirs('timit_train/wav')
		os.makedirs('timit_train/txt')
		os.makedirs('timit_train/phn')
		os.makedirs('timit_train/wrd')
	
	for file in temp:
		if(file.split('.')[-1] == 'wav'):
			name = "_".join(file.split('/')[2:])
			
	
			y, sr = librosa.load(file,16000)
			librosa.output.write_wav('timit_train/wav/{}'.format(name), y, sr)
			#os.system('cp {} timit_test/wav/{}'.format(file,name))
			os.system('cp {} timit_train/wav/{}'.format(file,name).replace('wav','wrd'))
			os.system('cp {} timit_train/wav/{}'.format(file,name).replace('wav','phn'))
			with open(file.replace('wav','txt'),'r') as f:
				lines = f.readlines()
				transcript = " ".join(lines[0].strip().split(' ')[2:])
			transcript = transcript.strip().upper()
			valid_punctuation = string.punctuation.replace("'","")
			transcript = transcript.translate(str.maketrans({a:None for a in valid_punctuation }))
			#print(transcript)
			with open('timit_train/txt/{}'.format(name).replace('wav','txt'),'w+') as w:
				w.write(transcript)
			m.write('/workspace/data/my_data/timit_train/wav/{}'.format(name) +','+'/workspace/data/my_data/timit_train/wav/{}'.format(name).replace('wav','txt') + '\n')
