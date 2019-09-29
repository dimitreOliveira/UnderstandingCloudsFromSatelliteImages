# Understanding Clouds from Satellite Images - Planning
 
## Working cycle:
1. Read competition information and relevant content to feel comfortable with the problem. Create a hypothesis based on the problem.
2. Initial data exploration to feel comfortable with the problem and the data.
3. Build the first implementation (baseline).
4. Loop through [Analyze -> Approach(model) -> Implement -> Evaluate].

## 1. Literature review (read some kernels and relevant content related to the competition).
- ### Relevant content:
  - [Medium post about the competition](https://towardsdatascience.com/sugar-flower-fish-or-gravel-now-a-kaggle-competition-8d2b6b3b118)
  - [Paper related about the competition research](https://arxiv.org/pdf/1906.01906.pdf)
  - [[Git] UNet-Segmentation-in-Keras-TensorFlow](https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb)
  - [[Git] unet for image segmentation](https://github.com/zhixuhao/unet)
  - [[Git] keras-unet](https://github.com/karolzak/keras-unet)
  - [[Git] segmentation_models](https://github.com/qubvel/segmentation_models)
  - [Losses for Image Segmentation](https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/)
  - [Thread abour Dice, F-Score and IOU](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou)

- ### Kernels:
  - [Segmentation in PyTorch using convenient tools](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools)
  - [Cloud: ConvexHull& Polygon PostProcessing (No GPU)](https://www.kaggle.com/ratthachat/cloud-convexhull-polygon-postprocessing-no-gpu)
 
- ### Insights:
  - #### Positive Insights
    - For patterns like 'Sugar' or 'Gravel' its not expected any specific orientation and rotation augmentation might be applied to the images as well.
  - #### Negative Insights
    - Fish-pattern often seems to be oriented in east-west-direction (right to left).
