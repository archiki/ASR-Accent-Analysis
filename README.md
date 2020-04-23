# Analyzing Confounding Effect of Accents in E-2-E ASR models

This reposiotry contains code for our paper ***How Accents Confound: Probing for Accent Information in End-to-End Speech Recognition Systems***, on understanding the confounding effect of accents in an end-to-end Automatic Speech Recognition (ASR) model: [DeepSpeech2](https://github.com/SeanNaren/deepspeech.pytorch) through several probing/analysis techniques, which is going to appear in [ACL 2020](acl2020.org).

## Requirements
* [Docker](https://docs.docker.com/engine/release-notes/): Version 19.03.1, build 74b1e89
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* apex==0.1
* numpy==1.16.3
* torch==1.1.0
* tqdm==4.31.1
* librosa==0.7.0
* scipy==1.3.1


## Instructions
1. Clone [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) and checkout the commit id `e73ccf6`. This was the stable commit used in all our experiments.
2. Use the docker file provided in this directory and build the docker image followed by running it via the bash entrypoint,use the commands below. This should be same as the dockerfile present in your folder deepspeech.pytorch, the instructions in the `README.md` of that folder have been modified. 
```
sudo docker build -t  deepspeech2.docker .
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --entrypoint=/bin/bash --net=host --ipc=host deepspeech2.docker
```
3. The additional and/or modified files can be found in `DeeSpeech/` along with our trained model and Language Model (LM) used in `DeepSpeech/models`.
4. Install the [Mozilla Common Voice Dataset](https://voice.mozilla.org/en/datasets), [TIMIT Dataset](https://catalog.ldc.upenn.edu/LDC93S1) used in the experiments and the optional [Librispeech Dataset](www.openslr.org/12/) which is used only for training purposes.
5. **Preparing Manifests**: The data used in [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) is required to be in *.csv* called *manifests* with two columns: `path to .wav file, path to .txt file`. The *.wav* file is the speech clip and the *.txt* files contain the transcript in upper case. For Librispeech, use the `data/librispeech.py` in [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch). For others, use the `DeepSpeech/make_{MCV,timit}_manifest.py` provided. The file corresponding to TIMIT works on the original folder structure whereas as for MCV, we need to provide a *.txt* file with entries of the format- `file.mp3 : reference text`.


## Citation


## Acknoledgements
This project uses code from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
