import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import resnet

torch.manual_seed(10)
# Seed is same as that used for training so that we obtain the same validation set as we got in training

##################################################################
################# Using the Resnet Architecture ##################

model = resnet.resnet(resnet.BasicBlock, [2,2,2,2], 256) # Resnet18
model.load_state_dict(torch.load('best_ckpt.pth'))

from dataloader import FaceDataset
from torch.utils.data import DataLoader

#################################################################
############## Creating trainloader and valloader ###############
dataset = FaceDataset()
a = int(len(dataset)*0.85)
b = len(dataset) - a
train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
dataloader = DataLoader(train_ds, batch_size=100, shuffle=True)
valloader = DataLoader(val_ds, batch_size=100, shuffle=False)
model = model.cuda()
model.eval()

def accuracy(real, fake):
        batch_size = real.shape[0]
        match = 0
        mismatch = 0
        for i in range(batch_size):
            for j in range(batch_size):
                product = torch.sum(real[i] * fake[j])
                real_mag = torch.norm(real[i],p=2,dim=0)
                fake_mag = torch.norm(fake[j],p=2,dim=0)
                similarity = product/(real_mag*fake_mag)
                if i == j:
                    if similarity.item() > thresh: # The similarity threshold for matching and mis-matching was found out experimentally
                        score = 1.0
                    else: score = 0.0
                    match += score
                else:
                    if similarity.item() > thresh:
                        score = 0.0
                    else: score = 1.0
                    mismatch += score
        return match/batch_size, mismatch/(batch_size*(batch_size - 1))

#################################################################
############# Comparing different threshold values ##############
# Threshold values are over a range of 0.5 to 1.0 with values spaced between 0.01 unit.
# The validation match and mis-match accuracies are found and plotted against the threshold value.
# Threshold value is chosen where the match and mis-match accuracy are the highest and are almost the same.

match_list = []
mis_match_list = []
for thresh in np.arange(0.5,1.0,0.01):
    print('Thresh_value: ',thresh)

    with torch.no_grad():
        running_acc = [0.,0.]
        for ref,aug in tqdm(valloader):
            ref, aug = ref.cuda(), aug.cuda()
            real = model(ref)
            fake = model(aug)
            match, mismatch = accuracy(real,fake)
            running_acc[0] += match
            running_acc[1] += mismatch
        print('Validation accuracy: ','Match: ',running_acc[0]/len(valloader),'Mismatch: ',running_acc[1]/len(valloader))
        match_list.append(np.round(running_acc[0]/len(valloader),4))
        mis_match_list.append(np.round(running_acc[1]/len(valloader),4))

# Plotting the accuracies vs threshold value graph
plt.plot(np.arange(0.5,1.0,0.01),match_list)
plt.plot(np.arange(0.5,1.0,0.01),mis_match_list)
plt.show()