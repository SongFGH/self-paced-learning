import torch
import torch.nn as nn
import torch.nn.functional as F

class CFCNN_transfer_sigmoid(nn.Module):
    def __init__(self, a):
        super(CFCNN_transfer_sigmoid, self).__init__()
        self.conv1 = a.features[0]
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = a.features[3]
        self.conv3 = a.features[6]
        self.conv4 = a.features[8]
        self.conv5 = a.features[10]
        
        self.fc6 = nn.Linear(256*6*6, 1024)
        self.fc7 = nn.Linear(1024*5, 1024)
        self.fc8 = nn.Linear(1024, 2)
        
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()
        
        self.convX1 = nn.Conv2d(64, 1024, kernel_size=27)
        self.convX2 = nn.Conv2d(192, 1024, kernel_size=13)
        self.convX3 = nn.Conv2d(384, 1024, kernel_size=6)
        self.convX4 = nn.Conv2d(256, 1024, kernel_size=6)
        
    def forward(self, x):
#        print(x)
        x = self.pool(F.relu(self.conv1(x)))
        x1 = F.relu(self.convX1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x2 = F.relu(self.convX2(x))
        x = self.conv3(x)
        x3 = F.relu(self.convX3(self.pool(x)))
        x = F.relu(x)
        x = self.conv4(x)
        x4 = F.relu(self.convX4(self.pool(x)))
        x = F.relu(x)
        x= self.pool(F.relu(self.conv5(x)))
        
        x = x.view(-1, 256*6*6)
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        
        y = torch.cat((x, x1.view(-1, 1024), x2.view(-1, 1024), 
                       x3.view(-1, 1024), x4.view(-1, 1024)), 1)        
        y = self.dropout(y)
        y = F.relu(self.fc7(y))
        y = self.sigmoid(self.fc8(y))
        
        return y
