# Files corresponding to Phone Probes

## Confusion Matrices
After training the phone-classifies on frame-level representations we look at the confusion matrices present in folder `ConfusionMatrices` and are generated using `Confusion Matrix.ipynb`. We observe that the accuracy trends are reflected in the confusion matrices as well. The confusion in the phones also appears to be *phonetically sound* and shows different patterns for different accents. Each folder in `ConfusionMatrices/` corresponds to different accents and contains the confusion matrices for all the layers.

### Confusion Matrix for Indian Accent
<img align="left" width="290" height="290" src=ConfusionMatrices/indian/Conf_spec_indian.png> <img align="center" width="290" height="290" src=ConfusionMatrices/indian/Conf_conv_indian.png><img align="right" width="290" height="290" src=ConfusionMatrices/indian/Conf_rnn_0_indian.png>

<img align="left" width="290" height="290" src=ConfusionMatrices/indian/Conf_rnn_1_indian.png> <img align="center" width="290" height="290" src=ConfusionMatrices/indian/Conf_rnn_2_indian.png><img align="right" width="290" height="290" src=ConfusionMatrices/indian/Conf_rnn_3_indian.png>
<p align="center">
  <img width="290" height="290" src=ConfusionMatrices/indian/Conf_rnn_4_indian.png>
</p>

### Confusion Matrix for Canadian Accent
<img align="left" width="290" height="290" src=ConfusionMatrices/canada/Conf_spec_canada.png> <img align="center" width="290" height="290" src=ConfusionMatrices/canada/Conf_conv_canada.png><img align="right" width="290" height="290" src=ConfusionMatrices/canada/Conf_rnn_0_canada.png>

<img align="left" width="290" height="290" src=ConfusionMatrices/canada/Conf_rnn_1_canada.png> <img align="center" width="290" height="290" src=ConfusionMatrices/canada/Conf_rnn_2_canada.png><img align="right" width="290" height="290" src=ConfusionMatrices/canada/Conf_rnn_3_canada.png>
<p align="center">
  <img width="290" height="290" src=ConfusionMatrices/canada/Conf_rnn_4_canada.png>
</p>











