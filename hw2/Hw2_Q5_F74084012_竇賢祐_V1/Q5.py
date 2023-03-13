from torchvision import datasets, transforms, models
import torchvision
from torchsummary import summary
from torch import nn
from torch import optim

import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import random
import numpy as np
from PIL import Image
import math


model = models.resnet50(pretrained=True)
for params in model.parameters():
    params.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 2),
                           nn.LogSoftmax(dim=1))

model.fc = classifier
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.fc.parameters(), lr=0.003)

# Moves the model to either CPU or GPU

model.load_state_dict(torch.load('./resnet_dog_cat.pth'))
#model.to('cuda')
model.eval()
#
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

test_data = datasets.ImageFolder('./PetImages/all', transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle = True)

def Q51():
    summary(model, input_size=(3, 224, 224), device='cpu')
    return


def Q52():
    img = cv2.imread('tensorboard.PNG')
    cv2.imshow('tensorboard', img)
    return


def Q53(num):
    # if (num == ""):
    #     num = 1
    # else:
    #     num = int(float(num))
    # count = 0
    for data, label in testloader:
        # count += 1
        # if count != num:
        #     continue
        # print(data)
        # print("----")
        # print(label)
        out = model(data)
        out = out.detach().numpy()

        data = torchvision.utils.make_grid(data)
        data = data / 2 + 0.5  # unnormalize
        data = data.numpy()
        plt.figure()
        plt.imshow(np.transpose(data, (1, 2, 0)))
        if out[0][0]>out[0][1]:
            plt.title("Class : Cat")
        elif out[0][0] < out[0][1]:
            plt.title("Class : Dog")
        else:
            plt.title("error")
        plt.show()
        # print(out)
        break
    return


def Q54():
    labels = ['before random erasing', 'after random erasing']
    acc = [99,95]
    x = np.arange(len(labels))
    plt.bar(x, acc, color=['blue', 'blue'])
    plt.xticks(x, labels)
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig("acc_random_erase.jpg")
    random_erase()
    return

def random_erase(sl = 0.2, sh = 0.4, rl = 0.2, rh = 0.4, p = 0.3):
    is_erase = 0.2
    original = cv2.imread('./PetImages/Cat/1.jpg')
    W = original.shape[0]
    H = original.shape[1]
    area = original.shape[0] * original.shape[1]
    cv2.imshow("original", original)
    if(is_erase < p):
        while(True):
            s = random.uniform(sl,sh) * area
            r = random.uniform(rl,rh)
            He = math.sqrt(s*r)
            We = math.sqrt(s/r)
            x = random.uniform(0,original.shape[0])
            y = random.uniform(0,original.shape[1])
            if(((x + We)<=W) and ((y + He)<=H)):
                for i in range(H):
                    for j in range(W):
                        if ((x <= j <= x+We) and (y <= i <= y+He)):
                            original[i][j] = random.randint(0,255)
                cv2.imshow("after", original)
                return
    else:
        cv2.imshow("after", original)
        return

