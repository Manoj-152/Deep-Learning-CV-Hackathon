import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import resnet

torch.manual_seed(10)
# Note that the seed should be the same in threshold_experimenation.py

##################################################################
################ Using the Resnet18 Architecture #################
#     ---> Main reason for using resnet is to solve the vanishing gradient problem (which can greatly affect learning for face recognition).
#     ---> Resnet18 architecture is used here. It contains a total of 8 residual blocks (BasicBlock). The model gives out 256d feature vector for a face image.

model = resnet.resnet(resnet.BasicBlock, [2,2,2,2], 256) # Resnet18

num_params = np.sum([p.nelement() for p in model.parameters()])
print(num_params, ' parameters')

from dataloader import FaceDataset
from torch.utils.data import DataLoader

#################################################################
############## Creating trainloader and valloader ###############
dataset = FaceDataset()
a = int(0.85*len(dataset))
b = len(dataset) - a
train_ds, val_ds = torch.utils.data.random_split(dataset, (a, b))
dataloader = DataLoader(train_ds, batch_size=100, shuffle=True)
valloader = DataLoader(val_ds, batch_size=100, shuffle=False)
model = model.cuda()

###################################################################
################ Creating a custom Loss function ##################

# Note: Here real refers to the tensor of reference images and fake refers to the tensor of selfie images of people in a batch
# Things taken into account in this loss function:
#     ---> The similarity between the reference image and selfie image of the same person are to be maximized.
#     ---> The similarity between the reference and selfie image of different persons (other people present in the batch) are to be minimized.
#
# Step 1: The real matrix and fake matrix are multiplied with each other or themselves to create real_real, fake_fake, fake_real and real_fake combinations.
# Step 2: The diagonal of real_fake and fake_real matrices contains the similarities between different images of the same person. So they are taken as positive logits
# Step 3: The non_diagonal elements of all the multiplied matrices contain similarities between different people and so they are taken under negative logits.
# Step 4: The elements are reshaped such to a (2N,2N-1) matrix such that the first column contains the positive logits.
# Step 5: The cross-entropy loss is then used with first class as the true class so that the positive logits are maximized and negative logits are minimized.

class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
             
    def forward(self, real, fake):
        batch_size = real.shape[0]
        mask = torch.ones((batch_size,batch_size),dtype=bool).fill_diagonal_(0)
        labels = torch.zeros(2*batch_size).long().cuda()\
        
        fake = F.normalize(fake,p=2,dim=-1)
        real = F.normalize(real,p=2,dim=-1)

        fake_transpose = fake.permute(1,0)
        real_fake_multi = torch.mm(real,fake_transpose) # Shape (N,N)
        fake_real_multi = torch.mm(fake,real.permute(1,0)) # Shape (N,N)
        real_real_multi = torch.mm(real,real.permute(1,0)) # Shape (N,N)
        fake_fake_multi = torch.mm(fake,fake.permute(1,0)) # Shape (N,N)

        # Positive Logits
        logits_real_fake_pos = real_fake_multi[torch.logical_not(mask)]
        logits_fake_real_pos = fake_real_multi[torch.logical_not(mask)]

        # Negative Logits
        logits_real_fake_neg = real_fake_multi[mask].reshape(batch_size,-1)
        logits_fake_real_neg = fake_real_multi[mask].reshape(batch_size,-1)
        logits_real_real_neg = real_real_multi[mask].reshape(batch_size,-1)
        logits_fake_fake_neg = fake_fake_multi[mask].reshape(batch_size,-1)

        pos = torch.cat((logits_real_fake_pos, logits_fake_real_pos), dim=0).unsqueeze(1) # Shape (2N,1)
        neg_1 = torch.cat((logits_real_real_neg, logits_real_fake_neg), dim=1) # Shape (N,2N-2)
        neg_2 = torch.cat((logits_fake_real_neg, logits_fake_fake_neg), dim=1) # Shape (N,2N-2)
        neg = torch.cat((neg_1,neg_2), dim=0) # Shape (2N,2N-2)

        logits = torch.cat((pos,neg), dim=1) # Shape (2N,2N-1)
        loss = F.cross_entropy(logits, labels)
        return loss

optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001)
loss_fn = Loss()

###################################################################
################## Writing the Accuracy function ##################
# Only the accuracy of matching is calculated here
# The accuracy for correctly detecting match and mis-match between images is calculated in evaluate.py

def accuracy(real, fake):
    batch_size = real.shape[0]
    score = 0
    for i in range(batch_size):
        product = torch.sum(real[i] * fake[i])
        real_mag = torch.norm(real[i],p=2,dim=0)
        fake_mag = torch.norm(fake[i],p=2,dim=0)
        similarity = product/(real_mag*fake_mag)
        score += similarity
    return score/batch_size    

print('Start Training')
os.makedirs('./models', exist_ok=True)

###################################################################
###################### Training the model #########################
best_epoch = 0
best_acc = 0.0
for epoch in range(500):
    running_loss = 0
    model.train()
    for ref,aug in tqdm(dataloader):
        ref = ref.cuda()
        aug = aug.cuda()
        optimizer.zero_grad()
        real = model(ref) # real refers to feature vector of reference image
        fake = model(aug) # fake refers to feature vector of selfie image
        loss = loss_fn(real,fake)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch:',epoch,' Loss:',running_loss/len(dataloader))  

    # Calculating the accuracy on the training dataset
    model.eval()
    with torch.no_grad():
        running_acc = 0
        for ref,aug in tqdm(dataloader):
            ref, aug = ref.cuda(), aug.cuda()
            real = model(ref)
            fake = model(aug)
            running_acc += accuracy(real,fake)
        print('Training accuracy: ',running_acc.item()/len(dataloader))

    # Calculating the accuracy on the validation dataset
    with torch.no_grad():
        running_acc = 0
        for ref,aug in tqdm(valloader):
            ref, aug = ref.cuda(), aug.cuda()
            real = model(ref)
            fake = model(aug)
            running_acc += accuracy(real,fake)
        print('Validation accuracy: ',running_acc.item()/len(valloader))

        # Writing the best validation accuracy model, but only after 50 epochs (giving time for the model to stabilize)
        if running_acc.item()/len(valloader) > best_acc and epoch>50:
            torch.save(model.state_dict(), 'best_ckpt.pth')
            best_acc = running_acc.item()/len(valloader)
            best_epoch = epoch
    if epoch%5 == 0: print('Best Accuracy: ',best_acc,'Epoch: ',best_epoch)
    
    # Writing every fifteenth model as .pth file in models directory
    if epoch%15 == 0: torch.save(model.state_dict(), './models/model-'+str(epoch)+'.pth')
