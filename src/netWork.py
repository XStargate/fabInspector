# netWork.py

import time

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models as models

class densenet161_mod(nn.Module):
    """
    The model based on densenet161
    """

    def __init__ (self):
        super(densenet161_mod, self).__init__()
        model = models.densenet161(pretrained=True)

        # set model's name
        self.module_name = str('densenet161_mod')

        self.features = model.features

        # set true if fix the parameter of densenet161
        """ for param in self.features.parameters():
            param.requires_grad = False """
        
        model.classifier = nn.Sequential(
            nn.Linear(2208, 1104),
            nn.ReLU(),
            nn.Linear(1104, 552),
            nn.ReLU(),
            nn.Linear(552, 11)
        )

        # initialize the classifier layer
        model.apply(self.__initialize_weights)
        
        self.model_mod = model

    def forward(self, x):
        x = self.model_mod(x)
        x = x.view(-1, 11)

        return x

    def __initialize_weights(self, m):

        # initialize the fc layer by normal distribution
        # if type(m) == nn.Linear:
        #     m.weight.data.normal_(0, 0.01)
        #     m.bias.data.zero_()

        # initialize the fc layer by xavier uniform method
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

class densenet161_base(nn.Module):
    """
    The splitting branch model based on the densenet161
    """
    def __init__(self):
        super(densenet161_base, self).__init__()

        self.module_name = str("densenet161_base")

        model = models.densenet161(pretrained=True)

        # set true if fix the parameter of densenet161
        """ for param in self.features.parameters():
            param.requires_grad = False """
        
        self.base = nn.Sequential(*list(model.children())[:-1])
        
        self.dropout = nn.Dropout(p=0.5)

        self.fc11 = nn.Linear(2208, 11)
        self.fc2  = nn.Linear(2208, 2)
        
    def forward(self, x):

        x = self.base(x)
        x = self.dropout(x)
        x = x.view(-1, 2208)
        x11 = self.fc11(x)
        x2  = self.fc2(x)

        return x11, x2

    def __initialize_weights(self, m):

        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

class resnet152_mod(nn.Module):

    def __init__(self):
        super(resnet152_mod, self).__init__()

        self.module_name = str("resnet152_mod")

        model = models.resnet152(pretrained=True)

        # set true if fix the parameter of resnet161
        """ for param in self.features.parameters():
            param.requires_grad = False """
        
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

        # initialize the classifier layer
        model.apply(self.__initialize_weights)
        
        self.model_mod = model

    def forward(self, x):
        x = self.model_mod(x)
        x = x.view(-1, 11)

        return x

    def __initialize_weights(self, m):
        # initialize the fc layer by normal dis
        # if type(m) == nn.Linear:
        #     m.weight.data.normal_(0, 0.01)
        #     m.bias.data.zero_()

        # initialize the fc layer by xavier uniform method
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

class resnet152_base(nn.Module):
    """
    The splitting branch model based on the resnet152
    """
    def __init__(self):
        super(resnet152_base, self).__init__()

        self.module_name = str("resnet152_base")

        model = models.resnet152(pretrained=True)

        # set true if fix the parameter of resnet161
        """ for param in self.features.parameters():
            param.requires_grad = False """
        
        self.base = nn.Sequential(*list(model.children())[:-1])
        
        self.dropout = nn.Dropout(p=0.5)

        self.fc11 = nn.Linear(2048, 11)
        self.fc2  = nn.Linear(2048, 2)
        
    def forward(self, x):

        x = self.base(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x11 = self.fc11(x)
        x2  = self.fc2(x)

        return x11, x2

    def __initialize_weights(self, m):

        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, path, name):
        if name is None:
            name = self.module_name
        torch.save(self.state_dict(), name)
        
        return name

class vgg16_mod(nn.Module):
    """
    The model based on vgg16
    """
    def __init__ (self):
        super(vgg16_mod, self).__init__()
        model = models.vgg16_bn(pretrained=True)

        self.module_name = str('vgg16_bn')
        self.features = model.features

        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 11)
        )

        # initialize the classifier layer
        model.apply(self.__initialize_weights)
        
        self.model_mod = model

    def forward(self, x):
        x = self.model_mod(x)
        x = x.view(-1, 11)

        return x

    def __initialize_weights(self, m):
        # initialize the fc layer by normal dis
        # if type(m) == nn.Linear:
        #     m.weight.data.normal_(0, 0.01)
        #     m.bias.data.zero_()

        # initialize the fc layer by xavier uniform method
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

class vgg16_base(nn.Module):
    """
    The splitting branch model based on vgg16
    """
    
    def __init__(self):
        super(vgg16_base, self).__init__()

        self.module_name = str("vgg16_base")

        model = models.vgg16_bn(pretrained=True)

        # set true if fix the parameter of vgg16
        """ for param in self.features.parameters():
            param.requires_grad = False """
        
        self.base = nn.Sequential(*list(model.children())[:-1])
        
        self.dropout = nn.Dropout(p=0.5)

        self.fc11 = nn.Linear(25088, 11)
        self.fc2  = nn.Linear(25088, 2)
        
    def forward(self, x):

        x = self.base(x)
        x = self.dropout(x)
        x = x.view(-1, 25088)
        x11 = self.fc11(x)
        x2  = self.fc2(x)

        return x11, x2

    def __initialize_weights(self, m):

        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

def test():

    model = vgg16_mod()
    params = model.state_dict()
    img = torch.rand(2, 3, 550, 600)
    img = Variable(img)
    output = model(img)
    print(output.size())

if __name__ == '__main__':
    test()
