# Changed/Modified files of DeepSpeech

## Overview
The file `test_attr.py` is used to calculate gradient attributions and can be run using the command:
``` 
python test_attr.py --model-path models/deepspeech_final.pth --test-manifest {desired manifest}.csv --cuda 
```
The file `test_contr.py` is used to calculate gradient contributions and can be run using the command:
``` 
python test_contr.py --model-path models/deepspeech_final.pth --test-manifest {desired manifest}.csv --cuda 
```
Both these files use `model_modifed.py` which stores the intermediate gradients required for the calculations. This model file should reflect in `utils.py`. For the purpose of storing the gradient attributions/contributions, the file `data/data_loader.py` is modified to output the file name/ unique ID in each batch. 

