# convolutional neural network with the architecture inspired by the classical LeNet-5 (LeCun et al., 1998).

import torch
import torch.nn as nn
        
# CREATE YOUR MODEL HERE
# gets as input an STM image [N, C, 128, 128]
# returns height prediction  [N, 1]

class HeightPrediction(nn.Module):
    def __init__(self):
        super(HeightPrediction, self).__init__()
        
        # Block 1 - three convolutional layers
        
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2 - three convolutional layers
        self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(40)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(40)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(40)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3 - one convolutional layer
        self.conv7 = nn.Conv2d(40, 60, kernel_size=3)
        self.bn7 = nn.BatchNorm2d(60)
        self.relu7 = nn.ReLU(inplace=True)

        # Block 4 - one convolutional layer
        self.conv8 = nn.Conv2d(60, 40, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(40)
        self.relu8 = nn.ReLU(inplace=True)

        # Block 5 - one convolutional layer
        self.conv9 = nn.Conv2d(40, 20, kernel_size=1)
        self.bn9 = nn.BatchNorm2d(20)
        self.relu9 = nn.ReLU(inplace=True)
        
        # Global Average Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=5)

        # Fully-connected layer
        self.fc = nn.Linear(20*6*6, 1)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool1(x)

        # Block 2
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.maxpool2(x)

        # Block 3
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        
        # Block 4
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        
        # Block 5
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return(x)     


net = HeightPrediction()
