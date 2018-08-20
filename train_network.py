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
def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.423, 0.439, 0.452])
    std = np.array([0.290, 0.269, 0.273])
    inp = std*inp+mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
def train_single_network(round, network):
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
    image_datasets = {x: myImagefolder(os.path.join(data_dir, x), data_transforms[x]) 
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=100, 
                                                  shuffle=True, 
                                                  num_workers=4) 
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    
    model = torch.load(network + '/' + network + '_' + str(int(round) - 1) + '.pkl')
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    num_epochs = 300
    
    bestupdate_epoch = 0
    best_acc = 0.0
    
    ratio = (int(round) - 1) * 0.02 + 0.3 # round 1 -- 36
    number_per_batch = int(ratio * 100)
    print('ratio and size: {} {}'.format(ratio, number_per_batch))
    
    for epoch in range(num_epochs):
        if epoch >= 100 and (epoch - bestupdate_epoch) >= 20:
            break
        print('-' * 30)
        print((network + ' Epoch {}/{}').format(epoch + 1, num_epochs))
        print('bestupdate_epoch atm: {}'.format(bestupdate_epoch))
        since = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                print('training')
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
                inputs, labels, filename = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    losslist=[]
                    for i in range(100):
                        losslist.append([criterion(outputs[i].unsqueeze(0), 
                                                  labels[i]).cpu().data.numpy()[0], i])
                    losslist.sort()                    
                    losslist = losslist[:number_per_batch]

                    inputs = inputs.cpu()
                    labels = labels.cpu()
                    
                    inputs_choose = Variable(torch.FloatTensor(number_per_batch, 3, 224, 224))
                    labels_choose = Variable(torch.LongTensor(number_per_batch))
                    filename_choose = []
                    for i in range(number_per_batch):
                        inputs_choose[i] = inputs[losslist[i][1]]
                        labels_choose[i] = labels[losslist[i][1]]
                        filename_choose.append(filename[losslist[i][1]])
                    
                    inputs_choose = inputs_choose.cuda()
                    labels_choose = labels_choose.cuda()
                    model.train(True) # only while calculating loss for update, open dropout
                    optimizer.zero_grad()
                    outputs_choose = model(inputs_choose)
                    loss_choose = criterion(outputs_choose, labels_choose)
                    loss_choose.backward()
                    optimizer.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase] 
            print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                    phase, epoch_loss, epoch_acc))
            if phase == 'train':
                with open(network + '/loss_acc_list_' + round + '.txt', 'a') as varfile:
                    varfile.write(str(epoch_loss) + '\t' + str(epoch_acc) + '\t')
            if phase == 'val':
                with open(network + '/loss_acc_list_' + round + '.txt', 'a') as varfile:
                    varfile.write(str(epoch_loss) + '\t' + str(epoch_acc) + '\n')
            
            if epoch >= 95 and phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                bestupdate_epoch = epoch
                print('best_acc update !!')
                if os.path.exists(network + '/' + network + '_' + round + '.pkl'):
                    os.remove(network + '/' + network + '_' + round + '.pkl')
                torch.save(model, network + '/' + network + '_' + round + '.pkl')
            
        time_elapsed = time.time() - since        
        print('This epoch training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        
train_single_network('1','resnet')                    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
