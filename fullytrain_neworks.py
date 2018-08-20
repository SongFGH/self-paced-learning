import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()

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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x])
                    for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=100, 
                                              shuffle=True, 
                                              num_workers=4)
                    for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
print(use_gpu)

def train_model(model, criterion, optimizer, scheduler, num_epochs=300):
    
    for epoch in range(num_epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                print('training')
                model.train(True)
                scheduler.step()
                # print lr
                for p_ft in optimizer.param_groups:
                    print('LR: {}'.format(p_ft['lr']))
                
            else:
                print('validating')
                model.train(False)
            
            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase] 
            print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                    phase, epoch_loss, epoch_acc))
            if phase == 'train':
                with open(path + 'loss_acc_list_0.txt', 'a') as varfile:
                    varfile.write(str(epoch_loss) + '\t' + str(epoch_acc) + '\t')
            if phase == 'val':
                with open(path + 'loss_acc_list_0.txt', 'a') as varfile:
                    varfile.write(str(epoch_loss) + '\t' + str(epoch_acc) + '\n')
            if (epoch+1) >= 100 and phase == 'val' and epoch % 20 == 19:
                torch.save(model, path + 'model_' + str(epoch + 1) + '.pkl')

        time_elapsed = time.time() - since        
        print('This epoch training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))



path = 'resnet/'
model_ft = torch.load(path + 'resnet_init.pkl')
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=300)

            




path = 'cfcnn/'
model_ft = torch.load(path + 'cfcnn_init.pkl')
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=300)
            
            
            
            
            
            
            
            
            
            
