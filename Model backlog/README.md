## Model backlog (list of the developed model and it's score)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **{metric}**.
- **Runtime** is the time in seconds that the kernel took to finish.
- **Pb Leaderboard** is the Public Leaderboard score.
- **Pv Leaderboard** is the Private Leaderboard score.

---

## Models

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Source|
|-----|-----|----------|--------------|--------------|------|
|1-UNet ResNet34 - Baseline|0.236|0.234|0.617|0.615|Kaggle|
|2-UNet ResNet34 - LR Warm Up|0.245|0.241|0.611|0.610|Kaggle|
|3-UNet ResNet34 - RAdam optimizer|0.218|0.222|0.528|0.519|Kaggle|
|4-UNet ResNet34 - Increased augmentation|0.220|0.218|0.509|0.500|Kaggle|
|5-UNet ResNet34 - Normalization|0.268|0.268|0.611|0.606|Kaggle|
|6-unet-resnet34-min-mask-size-tunning|0.297|0.292|0.528|0.261|Kaggle|
|7-unet-resnet34-no-scalling|0.219|0.218|0.255|???|Kaggle|
|8-unet-resnet34-best-params|0.266|0.265|0.605|0.604|Kaggle|
|9-unet-resnet18-best-params|0.219|0.218|0.550|0.549|Kaggle|
|10-unet-resnet18-threshold-tunning|0.593|0.592|0.595|0.590|Kaggle|
|11-unet-resnet18-224x224|0.634|0.620|0.619|0.616|Kaggle|
|12-unet-resnet18-cosine-learning-rate|0.642|0.623|0.632|???|Kaggle|
|13-unet-resnet18-cyclic-triangular-learning-rate|0.622|0.612|0.622|???|Kaggle|
|14-unet-resnet18-cyclic-triangular2-lr|0.627|0.615|0.621|???|Kaggle|
|15-unet-resnet18-cyclic-exp-range-lr|0.623|0.616|0.623|???|Kaggle|
|16-unet-resnet18-warmup-classifier-head|0.637|0.624|0.625|???|Kaggle|
|17-unet-resnet18-radam|0.713|0.604|0.601|???|Kaggle|
|18-unet-resnet152-radam|0.713|0.612|0.623|???|Kaggle|
|19-unet-resnet152-more-augmentation|0.640|0.620|0.624|???|Kaggle|
|20-unet-resnet152-clean-dataset|0.627|0.613|0.609|???|Kaggle|
|21-unet-densenet169-rotate|0.643|0.625|0.618|???|Kaggle|
|23-unet-densenet169-pt2|000|000|000|???|Colab|
|25-unet-densenet121-pt2|000|000|000|???|Colab|
|26-unet-densenet121-pt3|000|000|000|???|Colab|
|27-unet-densenet201|000|000|000|???|Colab|
|28-unet-densenet169-randomsizedcrop-256x384|0.648|0.637|0.637|???|Kaggle|
|29-unet-mobilenetv2|000|000|000|???|Colab|
|30-linknet-densenet169|000|000|000|???|Colab|
|31-[5-Fold]-ResNet18-Gamma_256x384 <br>- [Fold1] <br>- [Fold2] <br>- [Fold3] <br>- [Fold4] <br>- [Fold5]|0.673 <br>- 0.666<br>- 000<br>- 000<br>- 000<br>- 000|0.680 <br>- 0.678<br>- 000<br>- 000<br>- 000<br>- 000|0.642<br>- 0.639<br>- 000<br>- 000<br>- 000<br>- 000|???<br>- ???<br>- ???<br>- ???<br>- ???<br>- ???|Local|
|32-unet-resnext50|000|000|000|???|Local|
|33-unet-resnet34|000|000|000|???|Local|
|34-unet-efficientnetb4-256x384|0.642|0.641|0.636|???|Kaggle|
|35-unet-efficientnetb4-256x384-normalization|0.375|0.374|0.639|???|Kaggle|
|36-unet-efficientnetb4-256x384-warmup|0.454|0.453|0.597|???|Kaggle|
|37-fpnet-densenet169|000|000|000|???|Colab|
|38-unet-densenet169|000|000|000|???|Local|
|39-unet-densenet169 - lower LR|000|000|000|???|Local|
|40-UNet DenseNet169 - TTA|0.454|0.611|0.603|???|Kaggle|
|41-[5-Fold]unet-densenet169_256x384 <br>- [Fold1] <br>- [Fold2] <br>- [Fold3] <br>- [Fold4] <br>- [Fold5]|000 <br>- 0.658<br>- 000<br>- 000<br>- 000<br>- 000|000 <br>- 0.676<br>- 000<br>- 000<br>- 000<br>- 000|000<br>- 0.638<br>- 000<br>- 000<br>- 000<br>- 000|???<br>- ???<br>- ???<br>- ???<br>- ???<br>- ???|Colab|
|42-[5-Fold]unet-densenet169_384x480 <br>- [Fold1] <br>- [Fold2] <br>- [Fold3] <br>- [Fold4] <br>- [Fold5]|000<br>- 0.655<br>- 000<br>- 000<br>- 000<br>- 000|000<br>- 0.666<br>- 000<br>- 000<br>- 000<br>- 000|000<br>- 0.652<br>- 000<br>- 000<br>- 000<br>- 000|???<br>- ???<br>- ???<br>- ???<br>- ???<br>- ???|Colab|
|52-unet-resnet152|000|000|000|???|Local|
|53-unet-resnet50|000|000|000|???|Local|


## Inference

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Source|
|-----|-----|----------|--------------|--------------|------|
|2-seg-46densenet169|0.620|0.621|0.624|???|Kaggle|
|3-seg-47densenet169|0.671|0.644|0.641|???|Kaggle|
|4-seg-48densenet169|0.629|0.632|0.620|???|Kaggle|
|5-seg-22densenet169|0.654|0.643|0.637|???|Colab|
|6-seg-24densenet121|0.664|0.651|0.643|???|Colab|
|7-seg-49resnet18|0.677|0.629|0.516|???|Kaggle|
|8-seg-50resnet18|0.708|0.637|0.541|???|Kaggle|
|9-seg-5-fold-resnet18-256x384|0.673|0.680|0.642|???|Kaggle|
|10-seg-5-fold-4-41-unet-densenet169-256x384|0.681|0.692|0.645|???|Kaggle|
|11-seg-45-fold-0-unet-resnet169-256x384|0.666|0.678|0.639|???|Kaggle|
|12-seg-fold-0-unet-densenet169-256x384|0.658|0.676|0.638|???|Kaggle|
|13-seg-fold-0-unet-densenet169-384x480|0.655|0.666|0.652|???|Kaggle|
|15-seg-5-fold-41-unet-densenet169-256x384|0.677|0.681|0.647|???|Kaggle|
|16-seg-43-unet-resnet18-384x480|0.670|0.654|0.653|???|Kaggle|
|17-seg-44-linknet-resnet18-384x480|0.665|0.654|0.650|???|Kaggle|
|18-seg-45-pspnet-resnet18-384x480|0.643|0.641|0.645|???|Kaggle|
|19-seg-51-fpn-resnet18-384x480|0.664|0.659|0.653|???|Kaggle|
|20-seg-4-models-resnet18-384x480|0.670|0.659|0.654|???|Kaggle|
|21-seg-5-fold-2-42-unet-densenet169-384x480|0.658|0.670|0.652|???|Kaggle|
