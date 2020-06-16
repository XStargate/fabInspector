# cp_to_train.py

import os
import shutil
from tqdm import tqdm

source_pic_root = '../data/processed/data_train'
aug_pic_root = '../data/processed/data_train_aug'
target_pic_root = '../data/processed/data_train_final'
if os.path.exists(target_pic_root):
    shutil.rmtree(target_pic_root)
os.makedirs(target_pic_root)

#1# copy original images to target folder
for parent, _, files in os.walk(source_pic_root):
    for file in tqdm(files):
        if not file[-3:] == 'jpg':  # skip if not image
            continue
        temp_target_pic_root = os.path.join(target_pic_root, parent.split('/')[-1])
        if not os.path.exists(temp_target_pic_root):
            os.makedirs(temp_target_pic_root)
        shutil.copyfile(os.path.join(parent, file), os.path.join(temp_target_pic_root, file))

#2# copy augmented images to target folder 
for parent, _, files in os.walk(aug_pic_root):
    for file in tqdm(files):
        if not file[-3:] == 'jpg':  # skip if not image
            continue
        temp_target_pic_root = os.path.join(target_pic_root, parent.split('/')[-1])
        if not os.path.exists(temp_target_pic_root):
            os.makedirs(temp_target_pic_root)
        shutil.move(os.path.join(parent, file), os.path.join(temp_target_pic_root, file))
