# data_test_augment.py

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

from pdb import set_trace

def show_pic(img, bboxes=None):
    '''
    input:
        img: image array
        bboxes: the all bounding box lists of images with format [[x_min, y_min, x_max, y_max]....]
        names: the corresponding name for each box
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
    cv2.namedWindow('pic', 0)
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')

import shutil
from xml_helper import *
from tqdm import tqdm
from data_augment_method import data_augment

def data_aug(pic_source_path, xml_source_path, pic_target_path, aug_ob, aug_need_num=1, add_norm=False):
    '''
    input:
        pic_source_path : the root path of source images
        xml_source_path : the root path of xml files
        pic_target_path : the root path of target images
        aug_ob : the augmentation object
        aug_need_num : the number of images that needs to be augmentated
    '''

    for parent, _, files in os.walk(pic_source_path):
        # random.shuffle(files)
        for file in tqdm(files):
            if not file[-3:] == 'jpg':
                continue
            if not add_norm and parent.split('/')[-1] == 'norm':
                continue
            aug_num = 0                       # the current number of augmentation
            while aug_num < aug_need_num:
                aug_num += 1
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(xml_source_path, file[:-4]+'.xml')

                img = cv2.imread(pic_path)
                h,w,_ = img.shape

                if os.path.exists(xml_path):    # only defects have xml files
                    coords = parse_xml(xml_path)        # get box info with format [[x_min,y_min,x_max,y_max,name]]
                    coords = [coord[:4] for coord in coords]
                else:
                    coords = [[0,0,w,h]]  # if images having no box (i.e. normal), give a box of entire image (i.e. not crop)

                # show_pic(img, coords)    # original images

                aug_img, aug_bboxes = aug_ob.dataAugment(img, coords)

                # show_pic(aug_img, aug_bboxes)  # the augmentated images

                tmp_pic_target_path = os.path.join(pic_target_path, parent.split('/')[-1])
                if not os.path.exists(tmp_pic_target_path):
                    os.mkdir(tmp_pic_target_path)
                target_pic_path = os.path.join(temp_pic_target_path, file[:-4]+'_aug'+str(aug_num)+'.jpg')
                cv2.imwrite(target_pic_path, aug_img)

if __name__ == '__main__':

    # training data augmentation: crop, light, add noise, cutout
    data_train_aug = data_augment(rotation_rate=0,
                crop_rate=0.5, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0, cutout_rate=0.5)

    # test data augmentation: light, add noise, flip
    test_data_aug = data_augment(rotation_rate=0,
                crop_rate=0, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, cutout_rate=0)

    pic_source_path = '../data/processed/data_train'
    xml_source_path = '../data/raw/xml'

    print('===== training data augmentation =====')
    target_path = '../data/processed/data_train_aug'
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

    data_aug(pic_source_path, xml_source_path, target_path, data_train_, aug_need_num=2)

    # print('===== test data augmentation =====')
    # target_path = '../data/data_for_valid'
    # if os.path.exists(target_path):
    #     shutil.rmtree(target_path)
    # os.makedirs(target_path)

    # data_aug(pic_source_path, xml_source_path, target_path, test_data_aug, add_norm=True)
