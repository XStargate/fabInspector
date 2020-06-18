# data folder : store all raw and processed dataset

## raw : the raw dataset
./raw/pics/ : all fabric images, including both normal and defective fabric images.  

./raw/xml/ : xml files of defective fabric images, which contains the types and locations of fabric defects. Note that normal fabric images do not have corresponding xml files.

## processed : the processed dataset
./processed/data_hist/ : all fabric images processed by the histogram equalization algorithm.  

./processed/data_train/ : the 90% of original images from ./processed/data_hist/ as training dataset.  

./processed/data_test/ : the 10% of original images from ./processed/data_hist/ as test dataset.  

./processed/data_train_aug/ : the augmented images from ./processed/data_train/.  

./processed/data_train_final/ : the final training dataset, including all data from both ./processed/data_train/ and ./processed/data_train_aug/.  
