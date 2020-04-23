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
1. Follow the installation instructions for Docker given in [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch). Build the docker image followed by running it via the bash entrypoint,use the commands below:
```
sudo docker build -t  deepspeech2.docker .
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --entrypoint=/bin/bash --net=host --ipc=host seannaren/deepspeech.pytorch:latest
```

2. 

## Citation


## Acknoledgements
This project uses code from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
