# data_augment_method.py

import time 
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure

def show_pic(img, bboxes=None):
    '''
    input:
        img: the image array
        bboxes: the all bounding box lists of images with format [[x_min, y_min, x_max, y_max]....]
        names: the name of each box
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
    cv2.namedWindow('pic', 0)  # 1 is original image
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

class data_augment():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5, 
                crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, cutout_rate=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

    # add noise
    def _addNoise(self, img):
        '''
        input:
            img: image array
        output:
            the image array added with noise
        '''
        # random.seed(int(time.time())) 
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True)*255


    # change light
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)
        return exposure.adjust_gamma(img, flag)

    # mask out
    def _mask(self, img, bboxes, nholes=1, threshold=0.5, length=100):
        '''
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : the coordinates of bounding box
            nholes (int): Number of patches to cut out of each image.
            threshold (float): the threshold to mask out.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB are two boxes, and return iou
            boxB is bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # get h and w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape

        mask = np.ones((h,w,c), np.float32)

        for n in range(nholes):

            overlap = True    # judge if cut box is overlapped with bounding box too much

            while overlap:
                y = np.random.randint(h)
                x = np.random.randint(w)

                # numpy.clip(a, a_min, a_max, out=None), clip resctricts the tuple element between a_min and a_max
                # if it is greater than a_max, make it equal to a_max, if it is less than a_min, make it equal to a_min
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                overlap = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        overlap = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # rotation
    def _rotate(self, img, bboxes, angle=5, scale=1.):
        '''
        input:
            img: image array, (h, w, c)
            bboxes: a list includes all bounding boxes, in which every element is [x_min, y_min, x_max, y_max]
            angle: rotation angle
            scale: default is 1
        output:
            rot_img: the image array after rotation
            rot_bboxes: the boundingbox coordinate list after rotation
        '''
        #---------------------- Rotate images ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # deg to rad
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]

        # affine transform
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        #---------------------- correct bbox coordinates ----------------------
        # rot_mat is the final rotation matrix
        # obtain the four mid points of original bbox, and convert the four points to new coordinates after rotation
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))

            concat = np.vstack((point1, point2, point3, point4))

            concat = concat.astype(np.int32)

            # obtain coordinates after rotation 
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh

            # add into list
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # crop
    def _crop(self, img, bboxes):
        '''
        the cropped images should include all boxes
        input:
            img: image array
            bboxes: a list includes all bounding boxes, in which every element is [x_min, y_min, x_max, y_max]
        output:
            crop_img: the image array after crop
            crop_bboxes: the boundingbox coordinate list after crop
        '''
        #---------------------- crop ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # randomly extend the smallest box
        crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # gurantee not cross the boardlines
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        #---------------------- crop boundingbox ----------------------
        # the boundingbox coordinate after crop
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])

        return crop_img, crop_bboxes

    # shift
    def _shift(self, img, bboxes):
        '''
        the images after shift should contain all boxes
        input:
            img: image array
            bboxes: a list includes all bounding boxes, in which every element is [x_min, y_min, x_max, y_max]
        output:
            shift_img: the image array after shift
            shift_bboxes: the bounding box coordinate list after shift
        '''
        #---------------------- shift ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #---------------------- shift boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y])

        return shift_img, shift_bboxes

    # flip
    def _flip(self, img, bboxes):
        '''
        the image after flip should contain all boxes
        input:
            img: image array
            bboxes: a list includes all bounding boxes, in which every element is [x_min, y_min, x_max, y_max]
        output:
            flip_img: the image array after flip
            flip_bboxes: the bounding box coordinate list after flip
        '''
        # ---------------------- flip ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        # 0.5 chance horizontal flip; 0.5 chance vertical flip
        if random.random() < 0.5:
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon:
            flip_img =  cv2.flip(flip_img, -1)
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- change boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max])
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min])

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        '''
        data augmentation
        input:
            img: image array
            bboxes: the bounding box coordinates
        output:
            img: the image after augmentation
            bboxes: the corresponding box of images after augmentation
        '''
        change_num = 0  # the numebr of change
        visual = False
        # print('------')
        while change_num < 1:   # default at least one augmentation
            if random.random() < self.crop_rate:        #crop
                if visual:
                    print('crop') 
                change_num += 1
                img, bboxes = self._crop(img, bboxes)

            if random.random() < self.rotation_rate:    #ration
                if visual:
                    print('rotation') 
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate(img, bboxes, angle, scale)

            if random.random() < self.shift_rate:        #shift
                if visual:
                    print('shift')
                change_num += 1
                img, bboxes = self._shift(img, bboxes)

            if random.random() < self.change_light_rate: #change lightness
                if visual:
                    print('light')
                change_num += 1
                img = self._changeLight(img)

            if random.random() < self.add_noise_rate:    #add noise
                if visual:
                    print('add noise')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  #cutout
                if visual:
                    print('cutout')
                change_num += 1
                img = self._mask(img, bboxes, nholes=3, threshold=0.5, length=200)

            if random.random() < self.flip_rate:    #flip
                if visual:
                    print('flip')
                change_num += 1
                img, bboxes = self._flip(img, bboxes)
            # print('\n')
        # print('------')
        return img, bboxes
