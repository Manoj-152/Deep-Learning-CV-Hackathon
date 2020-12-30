import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

#########################################################################
########################### Extracting Faces ############################

# Haarcascade is used here to extract the faces from the pictures.
# Group pictures are neglected in this function, as it is not possible to find which face belongs to the person of our interest.
# If not able to detect a single face from the list of pictures given, None is returned.
# If faces are detected, a list of the faces detected is returned.

def face_extractor(list):
    out_list = []
    cnt = 0
    for pics in list:
        image = cv2.imread(pics)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
        if len(faces) == 1:
            cnt += 1
            for (x,y,w,h) in faces:
                face = image[y:y+h,x:x+w]
                if w>100 and h>100: resized = cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA)
                else: resized = cv2.resize(face, (100, 100), interpolation = cv2.INTER_CUBIC)
                out_list.append(resized)
    
    if cnt == 0: return None
    else: return out_list


#########################################################################
######################### Creating Dataloader ###########################

# Init Function:
#     ---> Folder Names of different people are extracted, and reference and selfie photos are read.
#     ---> If the face cannot be extracted from the reference pictures, they are replaced by the selfie faces extracted.
#     ---> Similarly if face cannot be extracted from the selfie pictures, they are replaced by the reference faces extracted.
#     ---> If faces cannot be extracted from both, the reference picture is directly passed as the reference face and the selfie face.
# 
# Getitem Function:
#     ---> One picture is taken randomly from the reference faces and selfie faces each of a person.
#     ---> The selfie face obtained is augmented while the reference face is not augmented.
#     ---> Both the face pictures are changed to torch tensors and normalized, and then passed out as output

class FaceDataset(Dataset):
    def __init__(self, root_dir='trainset'):
        people_list = glob(os.path.join(root_dir,'*/*'))
        self.face_pics = []
        for person in tqdm(people_list):
            ref_photos = glob(os.path.join(person,'*script*.jpg'))
            all_photos = glob(os.path.join(person,'*.jpg'))
            selfie_photos = [photo for photo in all_photos if photo not in ref_photos]

            ref_face = face_extractor(ref_photos)
            selfie_face = face_extractor(selfie_photos)
            if ref_face == None:
                if selfie_face != None: 
                    ref_face = selfie_face
                else:
                    ref_face = []
                    for i in ref_photos:
                        img = cv2.imread(i)
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        if img.shape[0]>100 and img.shape[1]>100: resized = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
                        else: resized = cv2.resize(img, (100, 100), interpolation = cv2.INTER_CUBIC)
                        ref_face.append(resized)
            if selfie_face == None:
                selfie_face = ref_face

            self.face_pics.append((ref_face,selfie_face))
        self._init_transform()
    
    def _init_transform(self):
        color_changer = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        rand_changer = transforms.Lambda(lambda x: color_changer(x) if random.random() < 0.8 else x)
        crop = transforms.CenterCrop(75)
        rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < 0.75 else x)
        self.selfie_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                rand_crop,
                                transforms.Resize((100,100)),
                                rand_changer,
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
                                ])
        self.reference_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
                                ])

    def __getitem__(self, index):
        refs, selfies = self.face_pics[index]
        i = np.random.randint(len(refs))
        ref_pic = refs[i]
        j = np.random.randint(len(selfies))
        selfie_pic = selfies[j]
        ref_pic = Image.fromarray(ref_pic).convert('RGB')
        selfie_pic = Image.fromarray(selfie_pic).convert('RGB')
        ref_pic = self.reference_transform(ref_pic)
        selfie_pic = self.selfie_transform(selfie_pic)
        return ref_pic, selfie_pic

    def __len__(self):
        return len(self.face_pics)

#########################################################################
############################ Main Function ##############################

if __name__ == "__main__":
    dataset = FaceDataset()
    dataloader = DataLoader(dataset,batch_size=64, shuffle=True)

    # visualizing the dataloader output
    ref,fake = next(iter(dataloader))
    print('Reference image size: ',ref.shape)
    print('Augmented image size: ',fake.shape)
    # reversing the normalisation for visualization purpose
    reverse_transform = transforms.Compose([transforms.Normalize(mean=[-1.,-1.,-1.], std=[2, 2, 2])])
    ref = reverse_transform(ref)
    fake = reverse_transform(fake)
    plt.imshow(ref[0].permute(1,2,0))
    plt.show()
    plt.imshow(fake[0].permute(1,2,0))
    plt.show()
