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
|1-UNet ResNet34 - Baseline|0.236|0.234|0.617|???|Kaggle|
|2-UNet ResNet34 - LR Warm Up|0.245|0.241|0.611|???|Kaggle|
|3-UNet ResNet34 - RAdam optimizer|0.218|0.222|0.528|???|Kaggle|
|4-UNet ResNet34 - Increased augmentation|0.220|0.218|0.509|???|Kaggle|
|5-UNet ResNet34 - Normalization|0.268|0.268|0.611|???|Kaggle|
|6-unet-resnet34-min-mask-size-tunning|0.297|0.292|0.528|???|Kaggle|
|7-unet-resnet34-no-scalling|0.219|0.218|0.255|???|Kaggle|
|8-unet-resnet34-best-params|0.266|0.265|0.605|???|Kaggle|
|9-unet-resnet18-best-params|0.219|0.218|0.550|???|Kaggle|
|10-unet-resnet18-threshold-tunning|0.593|0.592|0.595|???|Kaggle|
|11-unet-resnet18-224x224|0.634|0.620|0.619|???|Kaggle|
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
|28-unet-densenet169-randomsizedcrop-256x384|0.648|0.637|0.637|???|Kaggle|
|34-unet-efficientnetb4-256x384|0.642|0.641|0.636|???|Kaggle|
|35-unet-efficientnetb4-256x384-normalization|0.375|0.374|0.639|???|Kaggle|
|36-unet-efficientnetb4-256x384-warmup|0.454|0.453|0.597|???|Kaggle|
