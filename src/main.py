# main.py

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

from netWork import densenet161_mod as densenet
from netWork import resnet152_mod as resnet
from netWork import resnet152_base as resnet_b
from losses import FocalLoss

def train():
    train_dir = '../data/data_for_train_test'
    valid_dir = '../data/data_for_valid_test'

    transf = transforms.Compose([
        transforms.Resize((550, 600), interpolation=Image.ANTIALIAS),
        transforms.ToTensor()
    ])

    train_img = ImageFolder(train_dir, transform=transf)
    valid_img = ImageFolder(valid_dir, transform=transf)
    train_loader = DataLoader(train_img, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_img, batch_size=1, shuffle=True)


    model = resnet_b()

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

    # Evaluate the loss function and accuracy with initial model without training
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            train_pred11, train_pred2 = model(data[0].cuda())
            
            batch_loss11 = loss11(train_pred11, data[1].cuda())
            class2 = torch.tensor([1 if i==torch.tensor([10]) else 0 for i in data[1]])

            batch_loss2  = loss2(train_pred2, class2.cuda())

            train_acc += np.sum(
                np.argmax(train_pred11.cpu().data.numpy(), axis=1) == data[1].numpy())

            train_loss += 0.7 * batch_loss11.item() + 1.65 * batch_loss2.item()

        for i, data in enumerate(valid_loader):

            # Test data augmentation
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

            val_acc += np.sum(np.argmax(val_pred11_avg.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += 0.7 * batch_loss11_avg.item() + 1.65 * batch_loss2_avg.item()

        # print summary
        epoch = 0
        epoch_start_time = time.time()
        print('Accuracy and Loss before training')
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                    train_acc/train_img.__len__(), train_loss/train_img.__len__(), val_acc/valid_img.__len__(),
                    val_loss/valid_img.__len__()))

    # Train the model
    val_loss_last = 10000
    for epoch in range(num_epoch):
        epoch_start_time= time.time()
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.

        # Start training
        model.train()
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()
            train_pred11, train_pred2 = model(data[0].cuda())

            batch_loss11 = loss11(train_pred11, data[1].cuda())
            class2 = torch.tensor([1 if i==torch.tensor([10]) else 0 for i in data[1]])

            batch_loss2  = loss2(train_pred2, class2.cuda())
            batch_loss = 0.7 * batch_loss11 + 1.65 * batch_loss2

            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(
                np.argmax(train_pred11.cpu().data.numpy(), axis=1) == data[1].numpy())

            train_loss += batch_loss.item()

        # Evaluate the loss function and accuracy on test data
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):

                # Test data augmentation
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

                val_acc += np.sum(np.argmax(val_pred11_avg.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += 0.7 * batch_loss11_avg.item() + 1.65 * batch_loss2_avg.item()

            # print summary
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                train_acc/train_img.__len__(), train_loss/train_img.__len__(), val_acc/valid_img.__len__(),
                val_loss/valid_img.__len__()))

            if (val_loss > val_loss_last):
                print ('Old lr = ', lr, ' New lr = ', lr * 0.5)
                lr = lr * 0.5
                for group in optimizer.param_groups:
                    group['lr'] = lr 

            val_loss_last = val_loss
            # Save the model
            model_path = '../models/'
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 
                'train_acc': train_acc/train_img.__len__(), 'train_loss': train_loss/train_img.__len__(),
                'val_acc': val_acc/valid_img.__len__(), 'val_loss': val_loss/valid_img.__len__(),
                'optimizer': optimizer.state_dict(), 'loss11_alpha': loss11.alpha, 
                'loss11_gamma': loss11.gamma, 'loss2_alpha': loss2.alpha, 'loss2_gamma': loss2.gamma},
                model_path+model.module_name+'_'+str("{:02d}".format(epoch+1))+'.pth.tar')

if __name__ == '__main__':
    train()
