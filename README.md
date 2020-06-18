# FabInspector

FabInspector is an end-to-end model to indentify 10 different small obsucre fabric defects from the normal fabrics.

## Introduction and context
In the textile or fabric manufacturing industry, fabric defects caused by machine malfunction or broken yarn seriously impact the quality of product and lead to huge economic losses, which reduces prices of fabrics with losses reaching 45- 65%. Detecting fabric defects is one of the most iport processing steps for webbing quality control. Traditional manual inspection is a labor-intensive process. Manual inspection of fabric defects is unreliable as it is seriously affected by factors including the intensity of the lights, fatigue and the experience of the inspector, etc. According to feedback from the textile industry, manual inspection accuracy is merely 60 - 75%. Therefore, an automatic, fast and accurate fabric defect detector is an urgent need in fabric industry, which can help to save billions of dollars during the manufacturing process. Thanks to the advances in hardware technology and computer vision techniques as well as pattern recognition accuracy, the automatic visual inspection is becoming a promising and realistic solution for the detection of fabric defects. In this project, we propose an end-to-end network based on pretrained ResNet and DenseNet framework to detect 10 defferent common small obsucre fabric defects from the normal fabrics. Our experimental results reach more than 79% accuracy on our test data, which indicates our model is very promising to be applied in the fabric industry in future.

## Data
We collect more than 2000 fabric images as our dataset. There are two major difficulties and challenges in our dataset. On one hand, our dataset is very imbalanced, in which more than half images are normal fabrics, and rest images are belong to 10 different defects (Fig. 1). So number of images of each defect is quite small compared to the normal fabrics, which makes it harder to train models to classify these defects. On the other hand, the area of defects is very small in most defective fabrics. As shown in Fig. 2, more than 80% defects only have less than 1% area, which significantly increases the difficulty to detect the defects.

![data_imbalance](https://github.com/XStargate/insight_project/blob/master/pics/data_imbalance.png)
<center>Fig 1. the histogram of number of images for normal fabrics and 10 different kinds of defects</center>

![data_area](https://github.com/XStargate/insight_project/blob/master/pics/data_area.png)
<center>Fig 2. the area distribution of fabric defects. About 82% of defects have area less than 1%, while the area of about 17% of defects is between 1% and 10%. And the rest 1% of defects have area greater than 10%.</center>

## Data process
### Histogram equalization
Histogram equalization is a computer image processing technique used to improve contrast in images.It accomplishes this by effectively spreading out the most frequent intensity values, i.e. streatching out the intensity range of the image.  Histogram equalization is one of the best methods for image enhancement without loss of any information. Our most images have very skewed distribution on grayness and/or brightness, resulting in defects merged with normal background and hard to be identified. Therefore, the histogram equalization can significantly increase the contrast of most defect images (Fig. 3), which contributes a lot to improve the classification accuracy of our model.

![hist_equal](https://github.com/XStargate/insight_project/blob/master/pics/hist_equal.png)
<center>Fig 3. (a) the original images of defects. (b) the images of defects processed by the histogram equalization algorithm.</center>

### Data augmentation
Since the number of defective fabric images are very limited compared to normal fabrics. We implement a few data augmentation methods that are common and standard in the computer vision field to increase the dataset of defective fabrics, including flip, rotation, shift, crop, and slightly changing on grayness, brightness, etc.

## Framework / network

