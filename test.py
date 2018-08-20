import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class myImagefolder(datasets.ImageFolder):
    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path
    
data_transforms = {
        'train': transforms.Compose([
                transforms.Scale(240),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.423, 0.439, 0.452], [0.290, 0.269, 0.273])
        ]),
        'val': transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.423, 0.439, 0.452], [0.290, 0.269, 0.273])
        ]),
}
data_dir = 'traindata'
image_datasets = {x: myImagefolder(os.path.join(data_dir, x), 
                                          data_transforms[x])
                    for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=10, 
                                              shuffle=True, 
                                              num_workers=4)
                    for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

for data in dataloaders['train']:
    inputs, labels, filename = data
    inputs, labels = Variable(inputs), Variable(labels)
#    inputs = Variable(inputs.cuda())
#    labels = Variable(labels.cuda())
    break
    
def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.423, 0.439, 0.452])
    std = np.array([0.290, 0.269, 0.273])
    inp = std*inp+mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
imshow(inputs[9].cpu().data)
print(labels[9])
print(filename[9])

from cfcnn_model import CFCNN_transfer_sigmoid
path = 'cfcnn/'
m1 = torch.load(path + 'cfcnn_init.pkl')
m2 = torch.load(path + 'cfcnn_init.pkl')
#m=m.cuda()
optimizer1 = optim.SGD(m1.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer2 = optim.SGD(m2.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

#m.train(False)
#m(inputs[0].unsqueeze(0))    
    

