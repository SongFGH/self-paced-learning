import torch
import torch.nn as nn
from torchvision import models

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Sigmoid())
torch.save(model_ft,'RESNET/resnet_0.pkl')

from cfcnn_model import CFCNN_transfer_sigmoid

alexnet = models.alexnet(pretrained=True)
model_ft = CFCNN_transfer_sigmoid(alexnet)
torch.save(model_ft,'CFCNN/cfcnn_0.pkl')
