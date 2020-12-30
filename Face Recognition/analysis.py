import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
import numpy as np
import cv2
import resnet

###################################################################
############### Using the Resnet Architecture #####################

model = resnet.resnet(resnet.BasicBlock, [2,2,2,2], 256) # Resnet18
model = model.cuda()
model.load_state_dict(torch.load('best_ckpt.pth'))
model.eval()

###################################################################
############ Argparser to accept the input pictures ###############
parser = argparse.ArgumentParser(description='Enter the paths of the two photos')
parser.add_argument("pic1", help="Path of the first pic")
parser.add_argument("pic2", help="Path of the second pic")
args = parser.parse_args()

# A change in this face_extractor function is that the selfie image can contain multiple images, unlike the one we followed while training
def face_extractor(list):
    out_list = []
    cnt = 0
    for pics in list:
        image = cv2.imread(pics)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30)) # Please tune the scale-factor parameter to detect the face (needed sometimes)
        cnt += 1
        for (x,y,w,h) in faces:
            face = image[y:y+h,x:x+w]
            if w>100 and h>100: resized = cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA) # Resizing to 100x100 image size
            else: resized = cv2.resize(face, (100, 100), interpolation = cv2.INTER_CUBIC)
            out_list.append(resized)
    
    if cnt == 0: return None
    else: return out_list

print("Please note that the first photo(reference photo) must contain only a single face")
# Extracting the faces from the pictures. Throws an error if it is not able to.
face1 = face_extractor([args.pic1])
face2 = face_extractor([args.pic2])

if len(face1) > 1: print('Reference image contains more than one face. Please check.')
else:
    face1 = face1[0]
    face1_copy = face1.copy()
    plt.imshow(face1)
    plt.show()
    for i in range(len(face2)):
        plt.imshow(face2[i])
        plt.show()
    face2_copy = face2.copy()

    reference_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    face1 = reference_transform(face1)

    ###################################################################
    #### Checking similarities between reference and selfie images ####
    # Works on selfie images with multiple faces as well
    # But the reference image should have a single face

    face1 = face1.unsqueeze(0).cuda()
    similarities = []
    for i in range(len(face2)):
        face_2 = reference_transform(face2[i])
        face_2 = face_2.unsqueeze(0).cuda()
        real = model(face1)
        fake = model(face_2)
        real = real[0]
        fake = fake[0]

        product = torch.sum(real * fake)
        real_mag = torch.norm(real,p=2,dim=0)
        fake_mag = torch.norm(fake,p=2,dim=0)
        similarity = product/(real_mag*fake_mag)
        similarities.append(similarity.item())
    max_similarity = max(similarities)
    for i in range(len(similarities)):
        if similarities[i] == max_similarity:
            max_index = i

    ###################################################################
    #################### Printing the similarity ######################
    if len(face2) == 1:
        print("Similarity between faces: ",np.round(max_similarity,4))
    else:
        print("Maximum Similarity between faces",np.round(max_similarity,4))

    if max_similarity > 0.54:
        str1 = 'Match; ' + 'Similarity between faces: ' + str(np.round(max_similarity,4))
        print('Match')
    else:
        str1 = 'Mis-Match; ' + 'Similarity between faces: ' + str(np.round(max_similarity,4))
        print('Mis-Match')

    ###################################################################
    ####### Showing the matched faces along with the similarity #######

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(face1_copy)
    ax[1].imshow(face2_copy[max_index])
    fig.tight_layout()
    plt.suptitle(str1)
    plt.show()
