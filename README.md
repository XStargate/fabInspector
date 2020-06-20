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
Since the number of defective fabric images are very limited compared to normal fabrics. We implement a few data augmentation methods that are common and standard in the computer vision field to increase the training dataset of defective fabrics, including flip, rotation, shift, crop, and slightly changing on grayness, brightness, etc. Besides the augmentation on training dataset, we also augment the testing dataset by individual horizontal flip, individual vertical flip, and both horizontal and vertical flip simutaneously. Hence, each test image will become four images (including the original one) to be predicted/classified by our model, and the final prediction will be the average value from the predicting results of all four images. The augmentation on test dataset can be regards as a kind of model fusion, which turns out to increase the accuracy of our model on test dataset.

## Framework / network

We implement the transfer learning based on the pretrained ResNet152 and/or DenseNet161 models, and some other pretrained models like vgg16 and Inception-v4 can be also tried in our netwrok. The Fig. 4(a) shows the structure of our network, in which the processed images are input into the pretrained model and then extract feature of each channel by a global pooling layer. The key point in this network is to split the network into two branches, of which one is to do binary classification between normal and defective fabrics, and the other one is to classify 10 different kinds of fabrics from normal fabrics. The reason we have an additional branch of binary classification is to reduce the impact of dataset imbalance because the accumulative dataset from all defects has a comparable size with normal fabric dataset. Our experimental results indicate the binary classification is helpful to increase the accuracy of eventual classification on 11 classes.

We select the focal loss function as our loss function (Fig. 4(b)), which is defined in the following equation:

![focal_loss](https://github.com/XStargate/insight_project/blob/master/pics/focal_loss.png)

where ![p_t](https://github.com/XStargate/insight_project/blob/master/pics/p_t.png) is the classification probability; ![alpha_t](https://github.com/XStargate/insight_project/blob/master/pics/alpha_t.png) is the weights on different classes; ![gamma](https://github.com/XStargate/insight_project/blob/master/pics/gamma.png) is focusing parameter (default is 2) and ![p_t_gamma](https://github.com/XStargate/insight_project/blob/master/pics/p_t_gamma.png) is called modulating factor, which is to decrease the weights on classes that are easily classified, which increase the weight on difficult classifications. There are two major advantages of focal loss function compared to the cross entropy loss function. The first advantage is that it can mitigate the influence of dataset imbalance by adding more weights on small dataset manually, and the other advantage is it can automatically increase the weights on classes that hard to be classified correctly on the fly.

![network_loss](https://github.com/XStargate/insight_project/blob/master/pics/network_loss.png)
<center>Fig 4. (a) the network structure used in this project and the pretrained models we use are ResNet152 and DenseNet161. (b) the probability of ground truth of Focal loss function, which is the loss funtion used in this project (Tsung-Yi Lin et al. 2018).</center>

Since our network is splitted to two branches on both 2 classes and 11 classes, the loss is defined in the following equation:

![loss](https://github.com/XStargate/insight_project/blob/master/pics/loss.png)

where 70% of total loss is from the loss on 11 classifications and the rest 30% is from the loss on 2 classifications. We try a few different weights on those two types of classifications and turns out 70%/30% can achieve the best accuracy.

## Results and future works
In this project, we build the two models based on pretrained ResNet152 and DenseNet161 individually (Fig. 5), and both models can obtain the accuracy about 78% on the test data. Then we make a model fusion by combining both models, achieving more than 79% accuracy eventually. The accuracy is higher than common manual inspection accuracy between 50% and 65%, so we believe the model can help fabric manufacturing industry increase speed and accuracy on fabric defect detection. 

![accuracy](https://github.com/XStargate/insight_project/blob/master/pics/accuracy.png)
<center>Fig 5. the accuracy of model based on ResNet152 for each epoch, where the blue lines are accuracy of training dataset and red lines are accuracy of testing dataset.</center>

For this project, there are still a lot of other techniques worth exploring to increase the accuracy continuously in the future. For example, we can also try some other pretrained models, like vgg16, to do more model fusion. And we can also use some slided or cut windows to increase the size of dataset. Moreover, we can try to apply some object detection techniques on this project, like fast-RCNN, spatial pyramid pooling, Yolo network, etc.

## Code readme
Please see the readme under the ./src/ folder.
