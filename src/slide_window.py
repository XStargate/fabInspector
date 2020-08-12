# slide_window.py
"""
get the slided windows of defects
"""

import cv2
import math
import os

from cname2ename import cname_ename
from hist_equal import equal_hist
from xml_helper import parse_xml, generate_xml

from pdb import set_trace

class window_multi_size():

    def __init__(self, win_width, win_num_x = 6, win_num_y = 4, win_height=None):
        self.win_width = win_width
        if win_height is None:
            self.win_height = win_width
        else:
            self.win_height = win_height
        self.win_num_x = win_num_x
        self.win_num_y = win_num_y
        self.stride_x = int((2560 - self.win_width) / (self.win_num_x - 1))
        self.stride_y = int((1920 - self.win_height) / (self.win_num_y - 1))

    def _cal_iou(self, boxA, boxB):
        """
        input: boxA, window
               boxB, ground truth (defect) box
               the coordinates of boxA and boxB are
               [x_min, y_min, x_max, y_max]
        return: iou area
        """

        # determine the (x, y)-coordinates of the intersections rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        if xB <= xA or yB <= yA:
            return 0.0, 0.0;

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction adn ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # iou = interArea / float(boxAArea + boxBArea - interArea)
        iou_B = interArea / float(boxBArea)
        iou_A = interArea / float(boxAArea)

        # return the intersection over union value
        return iou_A, iou_B

    def cut_window(self, img, bboxes, threshold):

        """
        Input:
            img: image read from cv2
            bboxes: the list of bboxes
                    the coordinates of bboxes are
                    [xmin, ymin, xmax, ymax, name]
            threshold: the threshold of keep window
        Output:
            img_wins: the list of image windows
        """

        img_wins = []
        for i in range(self.win_num_x):
            for j in range(self.win_num_y):

                boxA_xmin = i * self.stride_x
                boxA_xmax = i * self.stride_x + self.win_width
                boxA_ymin = j * self.stride_y
                boxA_ymax = j * self.stride_y + self.win_height

                boxA = [boxA_xmin, boxA_ymin, boxA_xmax, boxA_ymax]

                # print ('boxA = ', boxA)

                defect_name = []

                for box in bboxes:
                    box_coord = box[0:4]
                    box_name = box[4]

                    iou_A, iou_B = self._cal_iou(boxA, box_coord)
                    if iou_B >= threshold or iou_A >= 0.5 or \
                       (box[3] <= boxA[3] and box[1] >= boxA[1] and
                        box[0] <= boxA[0] and box[2] >= boxA[2]) or \
                       (box[0] >= boxA[0] and box[2] <= boxA[2] and
                        box[1] <= boxA[1] and box[3] >= boxA[3]):
                        defect_name.append(box_name)

                # print ('defect_name = ', defect_name)
                if (len(list(set(defect_name))) == 1):
                    img_win = img[boxA_ymin:(boxA_ymax), boxA_xmin:(boxA_xmax)]
                    img_wins.append([img_win, defect_name[0]])
                    # print ('defect_name = ', defect_name[0])

        return img_wins

    def img_save(self, img_wins, img_name):

        for i in range(len(img_wins)):
            img_name_no_appendix = img_name[:-4]
            cv2.imwrite(img_name_no_appendix+'_'+str(i).zfill(2)+'.jpg', img_wins[i][0])


def win_train(class_train, pic_root_path, xml_root_path, pic_target_path, target_name_dict):

    multi_defects = ['J01_2018.06.19 09_16_36', 'J01_2018.06.13 13_53_54',
                     'J01_2018.06.22 10_19_57', 'J01_2018.06.17 15_07_25',
                     'J01_2018.06.25 15_17_24', 'J01_2018.06.28 15_10_04',
                     'J01_2018.06.26 13_28_17', 'J01_2018.06.27 08_49_31',
                     'J01_2018.06.19 10_25_31', 'J01_2018.06.26 15_52_33',
                     'J01_2018.06.28 14_51_12', 'J01_2018.06.17 14_37_37',
                     'J01_2018.06.17 09_28_58', 'J01_2018.06.22 15_23_41',
                     'J01_2018.06.28 14_34_05', 'J01_2018.06.22 09_03_24',
                     'J01_2018.06.17 14_37_09', 'J01_2018.06.22 10_24_03']

    win_width = [640, 800, 960]

    for width in win_width:

        # threshold = width / 1280
        threshold = 0.5
        window = window_multi_size(width)

        for key, value in class_train.items():

            if (key == 'norm'):

                for pic in value:

                    pic_path = os.path.join(pic_root_path, pic+'.jpg')
                    img = cv2.imread(pic_path, 0)

                    img_win = img[int((1920-width)/2):int((1920+width)/2),
                                  int((2560-width)/2):int((2560+width)/2)]

                    pic_target_folder = os.path.join(pic_target_path, key)
                    if not os.path.exists(pic_target_folder):
                        os.makedirs(pic_target_folder)

                    img_hist = equal_hist(img_win)

                    assert (len(img_hist) == width and len(img_hist[0]) == width), \
                        "Error: the window size is wrong for "+pic_path

                    cv2.imwrite(os.path.join(
                        pic_target_folder, pic+'_'+str(width)+'.jpg'), img_hist)

            else:

                for pic in value:

                    pic_path = os.path.join(pic_root_path, pic+'.jpg')
                    xml_path = os.path.join(xml_root_path, pic+'.xml')

                    img = cv2.imread(pic_path, 0)
                    coords = parse_xml(xml_path)

                    img_wins = window.cut_window(img, coords, threshold=threshold)

                    # check if img_wins is empty
                    assert img_wins, 'No winow for '+pic_path

                    if key not in multi_defects:

                        pic_target_folder = os.path.join(pic_target_path, key)
                        if not os.path.exists(pic_target_folder):
                            os.makedirs(pic_target_folder)
                        # pic_target_path_cur = os.path.join(pic_target_folder, pic+'.jpg')

                        img_hists = [[equal_hist(tmp[0]), tmp[1]] for tmp in img_wins]

                        for tmp in img_hists:
                            assert (len(tmp[0]) == width and len(tmp[0][0]) == width), \
                                "Error: the window size is wrong for "+pic_path

                        window.img_save(img_hists, os.path.join(
                            pic_target_folder, pic+'_'+str(width)+'.jpg'))

                    else:

                        for i in range(len(img_wins)):
                            tmp_name = cname_ename[img_wins[i][4]]
                            tmp_name = target_name_dict[tmp_name]

                            pic_target_folder = os.path.join(pic_target_path, tmp_name)
                            if not os.path.exists(pic_target_folder):
                                os.makedirs(pic_target_folder)

                            img_hist = equal_hist(img_wins[i][0])
                            assert (len(img_hist) == width and len(img_hist[0]) == width), \
                                "Error: the window size is wrong for "+pic_path

                            cv2.imwrite(os.path.join(
                                pic_target_folder, pic+'_'+str(width)+'_'+str(i).zfill(2)+'.jpg'),
                                        img_hist)

def win_test(class_test, pic_root_path, xml_root_path, pic_target_path, target_name_dict):

    win_width = [640, 800, 960]

    for width in win_width:

        pic_target_width_path = pic_target_path + '_' + str(width)

        for key, value in class_test.items():

            if (key == 'norm'):

                for pic in value:

                    pic_path = os.path.join(pic_root_path, pic+'.jpg')
                    img = cv2.imread(pic_path, 0)

                    img_win = img[int((1920-width)/2):int((1920+width)/2),
                                  int((2560-width)/2):int((2560+width)/2)]

                    pic_target_folder = os.path.join(pic_target_width_path, key)
                    if not os.path.exists(pic_target_folder):
                        os.makedirs(pic_target_folder)

                    img_hist = equal_hist(img_win)

                    assert (len(img_hist) == width and len(img_hist[0]) == width), \
                        "Error: the window size is wrong for "+pic_path

                    cv2.imwrite(os.path.join(
                        pic_target_folder, pic+'.jpg'), img_hist)

            else:

                for pic in value:

                    pic_path = os.path.join(pic_root_path, pic+'.jpg')
                    xml_path = os.path.join(xml_root_path, pic+'.xml')

                    img = cv2.imread(pic_path, 0)
                    coords = parse_xml(xml_path)

                    assert len(coords) == 1
                    tmp_name = cname_ename[coords[0][4]]
                    tmp_name = target_name_dict[tmp_name] if \
                        (tmp_name in target_name_dict) else 'defect_10'
                    assert tmp_name == key

                    defect_center_x = (coords[0][0] + coords[0][2]) / 2
                    defect_center_y = (coords[0][1] + coords[0][3]) / 2

                    xmin = int(defect_center_x - width / 2)
                    xmax = int(defect_center_x + width / 2)
                    ymin = int(defect_center_y - width / 2)
                    ymax = int(defect_center_y + width / 2)

                    if (xmin < 0):
                        xmax = width
                        xmin = 0
                    elif (xmax > 2560):
                        xmin = 2560 - width
                        xmax = 2560

                    if (ymin < 0):
                        ymax = width
                        ymin = 0
                    elif (ymax > 1920):
                        ymin = 1920 - width
                        ymax = 1920

                    assert (ymax > ymin and xmax > xmin and xmin >= 0 and
                            ymin >= 0 and xmax <= 2560 and ymax <= 1920)

                    img_win = img[ymin:ymax, xmin:xmax]

                    pic_target_folder = os.path.join(pic_target_width_path, key)
                    if not os.path.exists(pic_target_folder):
                        os.makedirs(pic_target_folder)

                    img_hist = equal_hist(img_win)
                    assert (len(img_hist) == width and len(img_hist[0]) == width), \
                        "Error: the window size is wrong for "+pic_path
                    cv2.imwrite(os.path.join(
                        pic_target_folder, pic+'.jpg'), img_hist)


def main(pic_root_path, xml_root_path, pic_target_path):

    win_width = [640, 800, 960]

    for width in win_width:

        threshold = width / 2580
        window = window_multi_size(win_width)

        for parent, _, files in os.walk(pic_root_path):

            for fname in files:

                xml_path = os.path.join(xml_root_path, fname)
                coords = parse_xml(xml_path)

                pics_path = os.path.join(pic_root_path, fname[:-3]+'jpg')
                img = cv2.imread(pics_path)

                # get the bboxes info from xml file
                # The coordinates of bboxes are
                # [xmin, ymin, xmax, ymax, name]

                # print (pics_path)
                img_wins = window.cut_window(img, coords, threshold=threshold)

                window.img_save(img_wins, os.path.join(pic_target_path, fname[:-3]+'jpg'))

if __name__ == '__main__':

    pic_root_path = '../data/pics_multi_defects/'
    xml_root_path = '../data/xml_multi_defects/'
    pic_target_path = '../data/pics_wins'

    main(pic_root_path, xml_root_path, pic_target_path)
