## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

#image size = (224, 224)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 5 5norm 3norm 3 3 3 fc fc
        # 5x5 square convolution kernel
        # convolutional layer output size = (W-F)/S + 1 = (224-5)/1 + 1 = 220
        # (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # batch normalisation applied to prevent overfitting
        #self.bn1 = nn.BatchNorm2d(32)
        # pooling layer output size = (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.2)

        # 5x5 square convolution kernel
        # convolutional layer output size = (110-5)/1 + 1 = 106
        # (64, 106, 106)
        self.conv2 = nn.Conv2d(32, 36, 5)
        #self.bn2 = nn.BatchNorm2d(64)
        # pooling layer output size = (64, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)
        # random dropout layer to prevent overfitting
        self.dropout2 = nn.Dropout(p=0.2)

        # 3x3 square convolution kernel
        # convolutional layer output size = (53-3)/1 + 1 = 51
        # (128, 51, 51)
        self.conv3 = nn.Conv2d(36, 48, 5)
        #self.bn3 = nn.BatchNorm2d(128)
        # pooling layer output size = (128, 25, 25)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p=0.2)

        # 3x3 square convolution kernel
        # convolutional layer output size = (25-3)/1 + 1 = 23
        # (256, 23, 23)        
        self.conv4 = nn.Conv2d(48, 64, 3)
        #self.bn4 = nn.BatchNorm2d(256)
        # pooling layer output size = (256, 11, 11)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p=0.2)
        
        # 3x3 square convolutional kernel
        # (23-3)/1 + 1 = 21
        # (512, 47, 47)
        self.conv5 = nn.Conv2d(64, 64, 3)
        # (512, 10, 10)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # 512 outputs * the 13*13 filtered/pooled map size        
        self.fc1 = nn.Linear(64*4*4, 136)
        #self.bn5 = nn.BatchNorm1d(1000)
        #self.dropout6 = nn.Dropout(p=0.2)

        #self.fc2 = nn.Linear(1000, 1000)
        #self.bn6 = nn.BatchNorm1d(1000)
        #self.dropout7 = nn.Dropout(p=0.2)

        #self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.dropout1(self.pool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.elu(self.conv4(x))))
        x = self.pool5(F.elu(self.conv5(x)))
        
        # flatten
        x = x.view(x.size(0),-1)

        #x = self.dropout6(F.elu(self.fc1(x)))
        #x = self.dropout7(F.elu(self.fc2(x)))
        x = self.fc1(x)

        return x