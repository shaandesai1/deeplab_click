
# coding: utf-8

import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
from new_loss import DiscriminativeLoss
import torchvision.models as models
from torch.nn import DataParallel
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from pycocotools.coco import COCO


from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.metrics import get_bin_map

writer = SummaryWriter()




model = DeepLab(num_classes=21,
                backbone='resnet',
                )

# model = DataParallel(model)
checkpoint = torch.load('deeplab-resnet.pth.tar',map_location='cpu')
state = model.state_dict()
state.update(checkpoint['state_dict'])
model.load_state_dict(state)


# Model
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('using gpus')
    model = DataParallel(model,device_ids=range(torch.cuda.device_count()))
#model.load_state_dict(torch.load('model.pth',map_location='cpu'))
model.to(device)


def my_collate(batch):
    img = [item['image'] for item in batch]
    mask = [item['label'] for item in batch]
    instance = [item['instances'] for item in batch]
    annid = [item['annid'] for item in batch]
    coord = [item['coords'] for item in batch]
    
    return [img,mask,instance,annid,coord]


#coco dataset training
train_df = COCOSegmentation(args, split='train', year='2017')
train_dataloader = DataLoader(train_df, collate_fn = my_collate, batch_size=32, shuffle=True)

val_df = COCOSegmentation(args, split='val', year='2017')
val_dataloader = DataLoader(val_df, collate_fn = my_collate, batch_size=32, shuffle=True)

data_dict = {'train':train_dataloader,'validation':val_dataloader}


# freeze all
for param in model.parameters():
    param.requires_grad = False

# unfreeze lastconv2
for name, child in model.named_children():
    if name == 'decoder':
        for n,c in child.named_children():
            if n == 'last_conv2':
                for param in c.parameters():
                    param.requires_grad = True

# Optimizer
parameters = model.parameters()

optimizer = optim.Adam(parameters, lr=1e-3)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                             mode='min',
    #                                             factor=0.1,
    #                                             patience=10,
#                                             verbose=True)

scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)


def train_model(model,optimizer,scheduler,num_epochs=10):
    #early = time.time()
    # Train
    n_iter_tr = 0
    n_iter_val = 0
    best_iou = -np.inf
    for epoch in range(num_epochs):
        #print(f'epoch : {epoch}')
        
        print('epoch')
        
        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()
            
            running_losses = 0.
            semantic_ious = 0.
            class_segment = np.zeros((21,2))
            
            for batched in data_dict[phase]:
                print('batch')
                images, sem_labels,instances,annid,coords = batched
                images = torch.stack(images)
                coords = torch.stack(coords)
                sem_labels = torch.stack(sem_labels)
                images = images.float().to(device)
                coords = coords.float().to(device)
                sem_labels = sem_labels.long().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                
                    inst_predict,sem_predict = model(images)
                    inst_predict = torch.cat([inst_predict,coords],dim=1)
#                    ce_loss = criterion_ce(sem_predict,sem_labels)
                    disc_loss =0.1*discriminative_loss(inst_predict,instances,annid,epoch)
                    
#                    ss = F.softmax(sem_predict,dim=1)
                    yp = torch.argmax(sem_predict,dim=1).cpu()
                    yt = sem_labels.cpu()
                    
                    loss = disc_loss
                    
                    tmp = 0.0
                    for i in range(images.size(0)):
                        ids = list(set(np.unique(yt[i,:,:])) -set([0,99]))
                        overlap = np.diag(get_bin_map(yt[i,:,:],yp[i,:,:].float()*torch.Tensor(np.where(yt[i,:,:]!=99,1.,0.))))
                        semantic_ious += np.mean(overlap)
                        tmp += np.mean(overlap)
                        class_segment[ids,0] += overlap
                        class_segment[ids,1] += 1
    
                    
   
                    if phase == 'train':
                        n_iter_tr += 1

                        loss.backward()
                        optimizer.step()
                        
                        writer.add_scalar('sem_iou_train_batch',tmp/images.size(0),n_iter_tr)
                        writer.add_scalar('CELoss_train_batch',ce_loss,n_iter_tr)
                        writer.add_scalar('DiscLoss_train_batch',disc_loss,n_iter_tr)
                    else:
                        n_iter_val += 1

                        writer.add_scalar('sem_iou_val_batch',tmp/images.size(0),n_iter_val)
                        writer.add_scalar('CELoss_val_batch',ce_loss,n_iter_val)
                        writer.add_scalar('DiscLoss_val_batch',disc_loss,n_iter_val)
                
          
                    running_losses += loss.cpu().data.tolist()[0]*images.size(0)
            
            avg_loss = running_losses/len(data_dict[phase].dataset)
            avg_iou = semantic_ious/len(data_dict[phase].dataset)
            avg_class_iou = class_segment[:,0]/(class_segment[:,1]+1e-8)
           
            if phase == 'train':
                writer.add_scalar('sem_iou_train_epoch',avg_iou,epoch)
                writer.add_scalar('Loss_train_epoch',avg_loss,epoch)
            
            else:
                writer.add_scalar('sem_iou_val_epoch',avg_iou,epoch)
                writer.add_scalar('Loss_val_epoch',avg_loss,epoch)
     
                #   scheduler.step(loss)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    print('Best Model!')
                    modelname = 'model.pth'
                    torch.save(model.state_dict(), modelname)


train_model(model,optimizer,scheduler,num_epochs=30)
