# DataAugForTrainAndTest.py

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
from DataAugmentForObjectDetection import DataAugmentForObjectDetection

def data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, aug_ob, need_aug_num=1, add_norm=False):
    '''
    input:
        source_pic_root_path : the root path of source images
        source_xml_root_path : the root path of xml files
        target_pic_root_path : the root path of target images
        aug_ob : the augmentation object
        need_aug_num : the number of images that needs to be augmentated
    '''

    for parent, _, files in os.walk(source_pic_root_path):
        # random.shuffle(files)
        for file in tqdm(files):
            if not file[-3:] == 'jpg':
                continue
            if not add_norm and parent.split('/')[-1] == 'norm':
                continue
            auged_num = 0                       # the current number of augmentation
            while auged_num < need_aug_num:
                auged_num += 1
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')

                img = cv2.imread(pic_path)
                h,w,_ = img.shape

                if os.path.exists(xml_path):    # only defects have xml files
                    coords = parse_xml(xml_path)        # get box info with format [[x_min,y_min,x_max,y_max,name]]
                    coords = [coord[:4] for coord in coords]
                else:
                    coords = [[0,0,w,h]]  # if images having no box (i.e. normal), give a box of entire image (i.e. not crop)

                # show_pic(img, coords)    # original images

                auged_img, auged_bboxes = aug_ob.dataAugment(img, coords)

                # show_pic(auged_img, auged_bboxes)  # the augmentated images

                temp_target_pic_root_path = os.path.join(target_pic_root_path, parent.split('/')[-1])
                if not os.path.exists(temp_target_pic_root_path):
                    os.mkdir(temp_target_pic_root_path)
                target_pic_path = os.path.join(temp_target_pic_root_path, file[:-4]+'_aug'+str(auged_num)+'.jpg')
                cv2.imwrite(target_pic_path, auged_img)

if __name__ == '__main__':

    # training data augmentation: crop, light, add noise, cutout
    dataAug_train = DataAugmentForObjectDetection(rotation_rate=0,
                crop_rate=0.5, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0, cutout_rate=0.5)

    # test data augmentation: light, add noise, flip
    dataAug_valid = DataAugmentForObjectDetection(rotation_rate=0,
                crop_rate=0, shift_rate=0, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, cutout_rate=0)

    source_pic_root_path = '../data/data_split_train'
    source_xml_root_path = '../data/xml'

    print('~~~~~~~~~~~~ training data augmentation ~~~~~~~~~~~~')
    target_pic_root_path = '../data/data_AugForTrain'
    if os.path.exists(target_pic_root_path):
        shutil.rmtree(target_pic_root_path)
    os.makedirs(target_pic_root_path)

    data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, dataAug_train, need_aug_num=2)

    # print('~~~~~~~~~~~~ test data augmentation ~~~~~~~~~~~~')
    # target_pic_root_path = '../data/data_for_valid'
    # if os.path.exists(target_pic_root_path):
    #     shutil.rmtree(target_pic_root_path)
    # os.makedirs(target_pic_root_path)

    # data_aug(source_pic_root_path, source_xml_root_path, target_pic_root_path, dataAug_valid, add_norm=True)
