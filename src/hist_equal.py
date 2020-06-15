# hist_equal.py

import numpy as np
import cv2 as cv
import os

from pdb import set_trace

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
            filenames.append(filename)

    return filenames, images

def write_images_to_folder(folder, filenames, imgs):
    assert len(filenames) == len(imgs)
    for i in range(len(imgs)):
        cv.imwrite(os.path.join(folder, filenames[i]), imgs[i])

    return None


def main():
    data_path = '../data/data_for_valid/defect_03/'
    filenames, images = load_images_from_folder(data_path)

    # select the method of histogram equalization (i.e. 'global' or 'regional')
    hist_method = 'global'

    equal_imgs = []
    for i in range(len(images)):

        if (hist_method == 'global'):
            res = cv.equalizeHist(images[i])

        elif (hist_method == 'regional'):
            clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(50, 50))
            res = clahe.apply(images[i])

        equal_imgs.append(res)

    output_path = '../data/tmp2/'
    write_images_to_folder(output_path, filenames, equal_imgs)
    
if __name__ == '__main__':
    main()
