# Models
### Pre-trained Model
The model used in all the experiments was trained from scratch using the training command:
```
python train.py --rnn-type lstm --hidden-size 1024 --hidden-layers 5  --train-manifest data/libri_train_manifest.csv --val-manifest data/libri_val_manifest.csv --epochs 60 --num-workers 16 --cuda  --learning-anneal 1.01 --batch-size 64 --
no-sortaGrad --opt-level O1 --loss-scale 1 --checkpoint --save-folder models/ --model-path models/deepspeech_final.pth 
```
**Dataset** | **WER** | **CER**
--- | --- | ---
Libri Clean |  | 
Libri Other |  | 

[Librispeech Pre-trained Model](https://drive.google.com/file/d/1njvgwduXkJXx3-0cHenL3-vfY5oTGzK3/view?usp=sharing)

### Language Model
