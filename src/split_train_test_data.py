# split_train_test_data.py
# split the dataset to training and test dataset

import os
import xml.etree.ElementTree as ET
from xml_helper import parse_xml
import random

from cname2ename import cname_ename
from slide_window import win_train, win_test

from pdb import set_trace


def get_name_from_xml(xml_path):

    coords = parse_xml(xml_path)
    name_list = {}

    for coord in coords:
        if coord[4] in name_list.keys():
            name_list[coord[4]] += (coord[2] - coord[0]) * (coord[3] - coord[1])
        else:
            name_list[coord[4]] = (coord[2] - coord[0]) * (coord[3] - coord[1])

    return max(name_list, key=name_list.get)


def split_11(pics_root_path, xml_root_path, target_name_dict):
    """
    Get the dictionary which splits all pics to 11 classes
    """

    classes = {'norm': [], 'defect_01': [], 'defect_02':[],
               'defect_03': [], 'defect_04': [], 'defect_05': [],
               'defect_06': [], 'defect_07': [], 'defect_08': [],
               'defect_09': [], 'defect_10': []}

    for parent, _, files in os.walk(pics_root_path):
        for fname in files:
            # pic_path = os.path.join(parent, fname)
            xml_path = os.path.join(xml_root_path, fname[:-4]+'.xml')

            if not os.path.exists(xml_path):
                tmp_name = '正常'
                classes['norm'].append(fname[:-4])
            else:
                tmp_name = get_name_from_xml(xml_path)

            tmp_name = cname_ename[tmp_name]

            if tmp_name in target_name_dict.keys():
                classes[target_name_dict[tmp_name]].append(fname[:-4])
            else:
                classes['defect_10'].append(fname[:-4])

    return classes


def split_train_test(class_all, test_exclude, test_pics_num):
    """
    Get the dictionary of train and test dataset
    """

    random.seed(2020)

    # class_train = {'norm': [], 'defect_01': [], 'defect_02':[],
    #                'defect_03': [], 'defect_04': [], 'defect_05': [],
    #                'defect_06': [], 'defect_07': [], 'defect_08': [],
    #                'defect_09': [], 'defect_10': []}

    # class_test = {'norm': [], 'defect_01': [], 'defect_02':[],
    #               'defect_03': [], 'defect_04': [], 'defect_05': [],
    #               'defect_06': [], 'defect_07': [], 'defect_08': [],
    #               'defect_09': [], 'defect_10': []}

    class_train = {}
    class_test = {}

    for key, value in class_all.items():

        fname_avail = [x for x in value if x not in test_exclude]

        class_test[key] = random.sample(fname_avail, test_pics_num[key])
        class_train[key] = [x for x in value if x not in class_test[key]]

    return class_train, class_test


def main():

    target_name_dict = {'norm':'norm', 'hole':'defect_01',
                        'bruise':'defect_02', 'tear':'defect_03',
                        'chafed':'defect_04', 'missing_pick':'defect_05',
                        'dropped_pick':'defect_06', 'end_out':'defect_07',
                        'filling_floats':'defect_08', 'stain' : 'defect_09'}

    test_pics_num = {'norm': 131, 'defect_01': 4, 'defect_02': 3,
                     'defect_03': 12, 'defect_04': 5,
                     'defect_05': 5, 'defect_06': 13,
                     'defect_07': 4, 'defect_08': 5, 'defect_09': 2,
                     'defect_10': 12}

    # get the test exclude list
    test_exclude_file = open('test_exclude_list', 'r')
    test_exclude_list = test_exclude_file.read().split('\n')
    test_exclude_file.close()

    # get the dictionary of 11 classes for all pics
    pic_root_path = '../data/pics/'
    xml_root_path = '../data/xml/'
    class_all = split_11(pic_root_path, xml_root_path, target_name_dict)

    # get the dictionary of training and test datset
    class_train, class_test = split_train_test(class_all, test_exclude_list, test_pics_num)

    # get the training pics
    # pic_train_target_path = '../data/data_win_train'
    # win_train(class_train, pic_root_path, xml_root_path, pic_train_target_path, target_name_dict)

    # get the test pics
    pic_test_target_path = '../data/data_win_test'
    win_test(class_test, pic_root_path, xml_root_path, pic_test_target_path, target_name_dict)


if __name__ == '__main__':
    main()
