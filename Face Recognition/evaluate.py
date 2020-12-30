import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import resnet
import argparse

##################################################################
################ Using the Resnet18 Architecture #################
model = resnet.resnet(resnet.BasicBlock, [2,2,2,2], 256) # Resnet18
model.load_state_dict(torch.load('best_ckpt.pth'))

num_params = np.sum([p.nelement() for p in model.parameters()])
print(num_params, ' parameters')

###################################################################
########### Argparser to accept the test dataset path #############
parser = argparse.ArgumentParser(description='Enter the path to the test set directory.')
parser.add_argument("path", help="Test set path")
args = parser.parse_args()

##################################################################
#################### Creating the valloader ######################
from dataloader import FaceDataset
from torch.utils.data import DataLoader
dataset = FaceDataset(args.path)
valloader = DataLoader(dataset, batch_size=100, shuffle=False)
model = model.cuda()

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
                if similarity.item() > 0.54: # The similarity threshold for matching and mis-matching was found out experimentally
                    score = 1.0
                else: score = 0.0
                match += score
            else:
                if similarity.item() > 0.54:
                    score = 0.0
                else: score = 1.0
                mismatch += score
    return match/batch_size, mismatch/(batch_size*(batch_size - 1))

##################################################################
################# Calculating validation score ###################
model.eval()
with torch.no_grad():
    running_acc = [0.,0.]
    for ref,aug in tqdm(valloader):
        ref, aug = ref.cuda(), aug.cuda()
        real = model(ref)
        fake = model(aug)
        match, mismatch = accuracy(real,fake)
        running_acc[0] += match
        running_acc[1] += mismatch
    # Accuracy for prediction of matching and mis-matching are printed
    print('Validation accuracy: ','Match: ',running_acc[0]/len(valloader),'Mismatch: ',running_acc[1]/len(valloader))