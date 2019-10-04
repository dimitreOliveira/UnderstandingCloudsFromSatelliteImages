## Model backlog (list of the developed model and it's score)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **{metric}**.
- **Runtime** is the time in seconds that the kernel took to finish.
- **Pb Leaderboard** is the Public Leaderboard score.
- **Pv Leaderboard** is the Private Leaderboard score.

---

## Models

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|
|-----|-----|----------|--------------|--------------|
|1-UNet ResNet34 - Baseline|0.236|0.234|0.617|???|
|2-UNet ResNet34 - LR Warm Up|0.245046|0.241565|0.611|???|
|3-UNet ResNet34 - RAdam optimizer|0.218|0.222|0.528|???|
|4-UNet ResNet34 - Increased augmentation|0.220|0.218|0.509|???|
|5-UNet ResNet34 - Normalization|0.268|0.268|0.611|???|
