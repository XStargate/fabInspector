# The code and file instructions

## src : all source codes 
hist_equal.py : process the dataset by histogram equlization algorithm. The process images are stored in the ./data/processed/data_hist/. The default setting is global histogram equalization, while the regional histogram equalization is also supportive.  

split_defects.py : split the normal fabric images and 10 different kinds of defective fabric images into different folders and extract 10% data as our test dataset.  

data_augment_method.py : a few different methods for data augmentation.  

data_test_augment.py : data augmentation on both training dataset and testing dataset.  

xml_helper.py : a helpler script to process xml files.  

cp_to_train.py : copy the original images from ../data/processed/data_train/ to ../data/processed/data_train_final/, and copy the augmented images from ../data/processed/data_train_aug/ to ../data/processed/data_train_final/.  

netWork.py : the networks used in this project.  

main.py : the main function for training the model.

model_fusion.py : combine a few different models and test it on the testing dataset.  

run.sh : a bash script to run all steps automatically.  

## new version with SeResNet101 and multi-scale windows
slide_window.py : get the multi-scale slided windows for original image.  

split_train_test_data.py : split the normal fabric images and 10 different kinds of defective fabric images into different folders and extract 10% data as our test dataset.  

main_se.py : the main function for training the model.
