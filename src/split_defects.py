# split_defects.py

'''
Split the raw data
'''

import os
import shutil
from tqdm import tqdm
from cname2ename import cname_ename
import xml.etree.ElementTree as ET
import random

out = open('./obj_repeat.txt','w')
def name_xml(xml_path, name_dict):
    '''
    Input:
        xml_path : xml files
        name_dict : the dict of folder names
    Output:
        the corresponding names of defects
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    names = set()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        names.add(name)

    names = list(names)

    final_name = names[0]
    repeat_cnt = 0
    for name in names:
        try:
            temp_name = cname_ename[name]
        except:
            print(name)
        if temp_name in name_dict.keys():
            repeat_cnt += 1
            final_name = name
    if repeat_cnt > 1:
        out.write(xml_path+'\n')
    return final_name


def split(pic_source_path, xml_source_path, pic_target_path, name_dict):
    '''
    Input:
        pic_source_path : the folder containing all source images
        xml_source_path : the folder containing all source xml files
        pic_target_path : the folder containing target images
        name_dict : the dict of target folder names
    '''

    for parent, _, files in os.walk(pic_source_path):
        for file in tqdm(files):
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(xml_source_path, file[:-4]+'.xml')

            if not os.path.exists(xml_path):            # normal iamges, cause only normal images has no xml file 
                temp_name = 'norm'
            else:
                temp_name = name_xml(xml_path, name_dict) # get the defect name of this image 

            temp_name = cname_ename[temp_name]

            if temp_name in name_dict.keys():
                target_pic_folder = os.path.join(target_path, name_dict[temp_name])
                if not os.path.exists(target_pic_folder):
                    os.makedirs(target_pic_folder)
                target_pic_path = os.path.join(target_pic_folder, file)
            else:
                target_pic_folder = os.path.join(target_path, 'defect_10')
                if not os.path.exists(target_pic_folder):
                    os.makedirs(target_pic_folder) 
                target_pic_path = os.path.join(target_pic_folder, file)
            shutil.copyfile(pic_path, target_pic_path)

def train_test_split(root_path, target_valid_root, ratio=0.1):
    '''
    Input:
        root_path : the path stores splitted training images
        target_valid_root : the path of validation images
        ratio : the ratio of validation images
    '''

    if os.path.exists(target_valid_root):
        shutil.rmtree(target_valid_root)

    for parent, _, files in os.walk(root_path):
        random.shuffle(files)   # shuffle the images
        limit = int(ratio*len(files))
        cnt = 0
        for file in files:
            cnt += 1
            if cnt > limit:
                break
            source_pic_path = os.path.join(parent, file)
            target_pic_path = os.path.join(target_valid_root+'/'+parent.split('/')[-1], file)

            if not os.path.exists(target_valid_root+'/'+parent.split('/')[-1]):
                os.makedirs(target_valid_root+'/'+parent.split('/')[-1])

            shutil.move(source_pic_path, target_pic_path)

if __name__ == '__main__':

    name_dict = {'norm':'norm', 'hole':'defect_01',
                 'bruise':'defect_02', 'tear':'defect_03',
                 'chafed':'defect_04', 'missing_pick':'defect_05',
                 'dropped_pick':'defect_06', 'end_out':'defect_07',
                 'filling_floats':'defect_08', 'stain' : 'defect_09'}

    pic_source_path = '../data/processed/data_hist'
    xml_source_path = '../data/raw/xml'
    target_path = '../data/processed/data_train'
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    split(pic_source_path, xml_source_path, target_path, name_dict)

    print('===== split train and test =====')
    train_test_split(target_path, '../data/processed/data_test', ratio=0.1)
