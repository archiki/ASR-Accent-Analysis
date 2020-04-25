# Models
### Pre-trained Model
The model used in all the experiments was trained from scratch using the training command:
```
python train.py --rnn-type lstm --hidden-size 1024 --hidden-layers 5  --train-manifest data/libri_train_manifest.csv --val-manifest data/libri_val_manifest.csv --epochs 60 --num-workers 16 --cuda  --learning-anneal 1.01 --batch-size 64 --
no-sortaGrad --opt-level O1 --loss-scale 1 --checkpoint --save-folder models/ --model-path models/deepspeech_final.pth 
```
Downlad Link: [Librispeech Pre-trained Model](https://drive.google.com/uc?export=download&id=1njvgwduXkJXx3-0cHenL3-vfY5oTGzK3)

### Language Model
We used the [4-gram ARPA Language Model](www.openslr.org/resources/11/4-gram.arpa.gz) for LM decoding which can be used by supplying the appropriate path to the argument `--lm-path` while running `test.py`.

### Performance

#### With Greedy Decoding
**Dataset** | **WER** | **CER**
--- | --- | ---
Libri Clean | 10.343 | 3.404
Libri Other | 28.865 | 12.319 

#### With LM based Decoding
**Dataset** | **WER** | **CER**
--- | --- | ---
Libri Clean | 5.955 | 2.489
Libri Other | 19.036 | 10.383 

