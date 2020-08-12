# main.py

import numpy as np
import random
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from netWork import densenet161_mod as densenet
from netWork import densenet161_base as densenet_b
from netWork import resnet152_mod as resnet
from netWork import resnet152_base as resnet_b
from netWork import se_resnet101_base as se_resnet_b
from losses import FocalLoss

from pdb import set_trace

class RandomRotation_FixedDegree(object):
    """ Randomly rotate the image at fixed angle
    """

    def __init__(self, degree, p=0.5, resample=False,
                 expand=False, center=None, fill=None):
        self.degree = degree
        self.p = p
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degree, p):

        k = random.randint(0, 1)
        return degree if k <= p else 0

    def __call__(self, img):

        angle = self.get_params(self.degree, self.p)

        return TF.rotate(img, angle, self.resample,
                         self.expand, self.center, self.fill)

def get_test_loss_pred(data, model, loss11, loss2):

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

    val_pred11_avg = (val_pred11+val_pred11_hor+
                      val_pred11_ver+val_pred11_hor_ver) / 4

    val_loss = 0.7 * batch_loss11_avg.item() + 1.65 * batch_loss2_avg.item()

    return val_pred11_avg, val_loss

def train():
    train_dir = '../data/data_win_train'
    # valid_dir_1 = '../data/data_for_valid_test_1'
    # valid_dir_2 = '../data/data_for_valid_test_2'
    # valid_dir_3 = '../data/data_for_valid_test_3'

    valid_dir_1 = '../data/data_win_test_640'
    valid_dir_2 = '../data/data_win_test_800'
    valid_dir_3 = '../data/data_win_test_960'

    transf_train = transforms.Compose([
        transforms.Resize((550, 550), interpolation=Image.ANTIALIAS),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotation_FixedDegree(90),
        RandomRotation_FixedDegree(180),
        RandomRotation_FixedDegree(270),
        transforms.ToTensor()
    ])

    transf_test = transforms.Compose([
        transforms.Resize((550, 550), interpolation=Image.ANTIALIAS),
        transforms.ToTensor()
    ])

    train_img = ImageFolder(train_dir, transform=transf_train)
    valid_img_1 = ImageFolder(valid_dir_1, transform=transf_test)
    valid_img_2 = ImageFolder(valid_dir_2, transform=transf_test)
    valid_img_3 = ImageFolder(valid_dir_3, transform=transf_test)

    train_loader = DataLoader(train_img, batch_size=10, shuffle=True)
    valid_loader_1 = DataLoader(valid_img_1, batch_size=16, shuffle=False)
    valid_loader_2 = DataLoader(valid_img_2, batch_size=16, shuffle=False)
    valid_loader_3 = DataLoader(valid_img_3, batch_size=16, shuffle=False)


    # model = densenet_b().cuda()
    # model = resnet_b().cuda()
    model = se_resnet_b().cuda()

    loss11 = FocalLoss(11, gamma=2)
    loss2  = FocalLoss(2, gamma=2)
    lr = 0.0005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                                momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    num_epoch = 60

    model.eval()
    train_acc = 0.
    train_loss = 0.
    val_acc = 0.
    val_loss = 0.

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            train_pred11, train_pred2 = model(data[0].cuda())
            """ ce_loss = F.cross_entropy(train_pred, data[1].cpu(), reduction='none')
            pt = torch.exp(-ce_loss)
            batch_loss = (0.25 * (1-pt)**2*ce_loss).mean() """

            batch_loss11 = loss11(train_pred11, data[1].cuda())
            class2 = torch.tensor([1 if j==torch.tensor([10]) else 0 for j in data[1]])

            """ if (data[1] != torch.tensor([10])):
                class2 = torch.tensor([0])
            else:
                class2 = torch.tensor([1]) """

            batch_loss2  = loss2(train_pred2, class2.cuda())

            train_acc += np.sum(
                np.argmax(train_pred11.cpu().data.numpy(), axis=1) == data[1].numpy())

            train_loss += 0.7 * batch_loss11.item() + 1.65 * batch_loss2.item()

        assert len(valid_loader_1) == len(valid_loader_2) and \
            len(valid_loader_1) == len(valid_loader_3)

        for i, (data1, data2, data3) in enumerate(
                zip(valid_loader_1, valid_loader_2, valid_loader_3)):

            assert torch.equal(data1[1], data2[1]) and torch.equal(data1[1], data3[1])
            val_pred11_avg_1, val_loss_1 =  \
                get_test_loss_pred(data1, model, loss11, loss2)

            val_pred11_avg_2, val_loss_2 =  \
                get_test_loss_pred(data2, model, loss11, loss2)

            val_pred11_avg_3, val_loss_3 =  \
                get_test_loss_pred(data3, model, loss11, loss2)

            for j in range(len(data1[1].numpy())):
                val_acc += np.sum(
                    np.argmax(val_pred11_avg_1.cpu().data.numpy()[j]) == data1[1][j].numpy() or
                    np.argmax(val_pred11_avg_2.cpu().data.numpy()[j]) == data2[1][j].numpy() or
                    np.argmax(val_pred11_avg_3.cpu().data.numpy()[j]) == data3[1][j].numpy())

            val_loss += (val_loss_1 + val_loss_2 + val_loss_3)

        # print summary
        epoch = 0
        epoch_start_time = time.time()
        print('Accuracy and Loss before training')
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch, num_epoch, time.time()-epoch_start_time, \
                    train_acc/train_img.__len__(), train_loss/train_img.__len__(), val_acc/valid_img_1.__len__(),
                    val_loss/valid_img_1.__len__()))

    val_loss_last = 10000
    for epoch in range(num_epoch):
        epoch_start_time= time.time()
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.

        model.train()
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()
            train_pred11, train_pred2 = model(data[0].cuda())

            batch_loss11 = loss11(train_pred11, data[1].cuda())
            class2 = torch.tensor([1 if j==torch.tensor([10]) else 0 for j in data[1]])

            batch_loss2  = loss2(train_pred2, class2.cuda())
            batch_loss = 0.7 * batch_loss11 + 1.65 * batch_loss2

            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(
                np.argmax(train_pred11.cpu().data.numpy(), axis=1) == data[1].numpy())

            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, (data1, data2, data3) in enumerate(
                zip(valid_loader_1, valid_loader_2, valid_loader_3)):

                assert torch.equal(data1[1], data2[1]) and torch.equal(data1[1], data3[1])

                val_pred11_avg_1, val_loss_1 =  \
                    get_test_loss_pred(data1, model, loss11, loss2)

                val_pred11_avg_2, val_loss_2 =  \
                    get_test_loss_pred(data2, model, loss11, loss2)

                val_pred11_avg_3, val_loss_3 =  \
                    get_test_loss_pred(data3, model, loss11, loss2)

                for j in range(len(data1[1].numpy())):
                    val_acc += np.sum(
                        np.argmax(val_pred11_avg_1.cpu().data.numpy()[j]) == data1[1][j].numpy() or
                        np.argmax(val_pred11_avg_2.cpu().data.numpy()[j]) == data2[1][j].numpy() or
                        np.argmax(val_pred11_avg_3.cpu().data.numpy()[j]) == data3[1][j].numpy())

                val_loss += (val_loss_1 + val_loss_2 + val_loss_3)

            if (val_loss > val_loss_last):
                for group in optimizer.param_groups:
                    group['lr'] = lr * 0.5

            val_loss_last = val_loss

            # print summary
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_acc/train_img.__len__(), train_loss/train_img.__len__(), val_acc/valid_img_1.__len__(),
                val_loss/valid_img_1.__len__()))

            # Save the model
            model_path = '../models/'
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 
                'train_acc': train_acc/train_img.__len__(), 'train_loss': train_loss/train_img.__len__(),
                'val_acc': val_acc/valid_img_1.__len__(), 'val_loss': val_loss/valid_img_1.__len__(),
                'optimizer': optimizer.state_dict(), 'loss11_alpha': loss11.alpha, 
                'loss11_gamma': loss11.gamma, 'loss2_alpha': loss2.alpha, 'loss2_gamma': loss2.gamma},
                model_path+model.module_name+'_'+str("{:02d}".format(epoch+1))+'.pth.tar')


if __name__ == '__main__':
    train()
