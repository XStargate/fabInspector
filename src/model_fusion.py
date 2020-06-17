# model_fusion.py
# This script is to load the model and do model fusion

import os
from os import listdir
from os.path import isfile, join
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

from netWork import densenet161_base as densenet_b
from netWork import resnet152_base as resnet_b
from losses import FocalLoss

from pdb import set_trace

def load_model_dict(path, model):

    """
    Load the model dictionary
    Input:
        path: the path of model dict
        model: the model
    Output:
        the model with loaded model dict
        the model dict
    """

    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    model_dict = model.state_dict()

    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)

    set_trace()

    return model, checkpoint

def model_combine(models, data, loss11, loss2):
    """
    Evaluate test data by the combined model
    Input:
        models: the model list
        data: the test data
        loss11: the loss function on 11 classes
        loss2: the loss function on 2 classes
    Output:
        the accuracy of combined model on test data
        the loss of combine model on test data
    """

    val_preds = []
    val_losses = []

    for model in models:

        val_pred11, val_pred2 = model(data[0].cuda())
        val_pred11_hor, val_pred2_hor = model(torch.flip(data[0], [3]).cuda())
        val_pred11_ver, val_pred2_ver = model(torch.flip(data[0], [2]).cuda())
        val_pred11_hor_ver, val_pred2_hor_ver = model(torch.flip(data[0], [2,3]).cuda())

        batch_loss11 = loss11(val_pred11, data[1].cuda())
        batch_loss11_hor = loss11(val_pred11_hor, data[1].cuda())
        batch_loss11_ver = loss11(val_pred11_ver, data[1].cuda())
        batch_loss11_hor_ver = loss11(val_pred11_hor_ver, data[1].cuda())

        batch_loss11_avg = \
            (batch_loss11+batch_loss11_hor+batch_loss11_ver+batch_loss11_hor_ver)/4

        class2 = torch.tensor([1 if i==torch.tensor([10]) else 0 for i in data[1]])

        batch_loss2  = loss2(val_pred2, class2.cuda())
        batch_loss2_hor = loss2(val_pred2_hor, class2.cuda())
        batch_loss2_ver = loss2(val_pred2_ver, class2.cuda())
        batch_loss2_hor_ver = loss2(val_pred2_hor_ver, class2.cuda())

        batch_loss2_avg = \
            (batch_loss2+batch_loss2_hor+batch_loss2_ver+batch_loss2_hor_ver)/4

        val_pred11_avg = (val_pred11+val_pred11_hor+val_pred11_ver+val_pred11_hor_ver) / 4
        val_loss = 0.7 * batch_loss11_avg.item() + 1.65 * batch_loss2_avg.item()

        val_preds.append(val_pred11_avg.cpu().data.numpy())
        val_losses.append(val_loss)

    val_acc_avg = np.average(np.array(val_preds), axis=0)
    val_acc = np.sum(np.argmax(val_acc_avg, axis=1) == data[1].numpy())

    val_loss_avg = np.average(np.array(val_losses))

    return val_acc, val_loss_avg


def main():

    valid_dir = '../data/processed/data_test/'

    transf = transforms.Compose([
        transforms.Resize((550, 600), interpolation=Image.ANTIALIAS),
        transforms.ToTensor()
    ])

    valid_img = ImageFolder(valid_dir, transform=transf)
    valid_loader = DataLoader(valid_img, batch_size=8, shuffle=True)

    loss11 = FocalLoss(11, gamma=2)
    loss2  = FocalLoss(2, gamma=2)

    model_path = '../models/'

    model_dicts = [f for f in listdir(model_path) if isfile(join(model_path, f))]

    models = []
    for i in range(len(model_dicts)):
        model_dict_path = join(model_path, model_dicts[i])
        model = resnet_b().cuda()
        model_trained, checkpoint = load_model_dict(model_dict_path, model)

        models.append(model_trained)

    val_acc = 0.
    val_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            val_a, val_l = model_combine(models, data, loss11, loss2)

            val_acc += val_a
            val_loss += val_l

        print('Val Acc: %3.6f loss: %3.6f' % (val_acc/valid_img.__len__(), val_loss/valid_img.__len__()))

if __name__ == '__main__':
    main()
