import torch
import torchvision
import matplotlib.pyplot as plt 
import random
import torch.nn as nn
from torchvision import transforms, models
from torchsummary import summary
import glob

import torchvision.transforms as transforms

import numpy as np
batch_size = 200
LR = 0.001
opti = "SGD"

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  # num_classes
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调 #((512 * 7 * 7, 512))
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),  #(512,128)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

model5 = VGGNet()
model5.load_state_dict(torch.load('./VGG16_2.pth'))
model5.eval()

def Q51():

    data_batch1 = trainset

    plt.figure()
    for i in range(9):

        ran = random.randint(0,50000)
        img, label = data_batch1[ran]
        img = torchvision.utils.make_grid(img)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.subplot(3,3,i+1)
        plt.title(classes[label])
        plt.xticks(color='w')
        plt.yticks(color='w')
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    #plt.imshow(img)
    plt.show()

def Q52():
    print("hyperparameters:")
    print("batch size: {}".format(batch_size))
    print("learning rate: {}".format(LR))
    print("optimizer: {}".format(opti))

def Q53():
    model = VGGNet()
    summary(model, input_size=(3, 32, 32), device='cpu')

def Q54():
    img = plt.imread("answer2.png")
    plt.imshow(img)
    plt.xticks(color='w')
    plt.yticks(color='w')
    plt.show()

def Q55(text):
    if (text!=""):

        img, label = testset[int(float(text))]
        out = model5(img.reshape(1, 3, 32`, 32))

        out = out.detach().numpy()

        for i in range(len(out[0])):
            if (out[0][i]< 0):
                out[0][i] = 0
        sum = out[0].sum()
        for i in range(len(out[0])):
            out[0][i] = out[0][i]/sum

        img = torchvision.utils.make_grid(img)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()

        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        plt.figure()
        x = np.arange(len(classes))
        plt.bar(x, out[0])
        plt.xticks(x,classes)
        plt.show()
