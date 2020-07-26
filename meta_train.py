# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:00:45 2019

@author: WHX
"""
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import DataLoader


from data_generate import DatasetFromFolder
from model import SPCNNet


def PSNR_value(mse):
    return 20 * log10(1/mse)

def train_task(model,task_num):
    train_set = DatasetFromFolder('data/meta_train/task'+str(task_num), upscale_factor=3, input_transform=transforms.ToTensor(),
    target_transform=transforms.ToTensor())
    trainloader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=False)
    init_parameters = model.parameters()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for i,data in enumerate(trainloader,0):
        inputs,targets=data
        optimizer.zero_grad()
        outputs = model(inputs)
        cost = criterion(outputs,targets)
        cost.backward()
        optimizer.step()
        print('MSEcost after '+str(i+1)+' picture(s): '+ str(cost.item()))
        print('PSNR after ' + str(i+1)+' picture(s): '+ str(PSNR_value(cost.item())))
    
    test_set = DatasetFromFolder('data/meta_train/task'+str(task_num), upscale_factor=3, input_transform=transforms.ToTensor(),target_transform=transforms.ToTensor())
    testloader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)
    loss = 0.0
    optimizer.zero_grad()
    for test in testloader:
        inputs,targets = test
        outputs = model(inputs)
        loss+=criterion(outputs,targets)
        
    loss.backward()
    lr = 0.001
    for f in init_parameters:
        f.data.sub_(f.grad.data*lr)
        
    for i,f1 in enumerate(init_parameters,0):
        for j,f in enumerate(model.parameters(),0):
            if i == j:
                f.data = f1.data.clone()
    
    print('task'+str(task_num)+' trained!')
    
def main():
    SPCNN = SPCNNet(3)
    for j in range(1,5):
        print('epoch number ------------ '+ str(j))
        for i in range(1,6):
            train_task(SPCNN,i)
    
    torch.save(SPCNN.state_dict(), 'meta_epochs/meta_trained_model')
        
        
if __name__ == "__main__":
    main()
        
        
        
    
        
