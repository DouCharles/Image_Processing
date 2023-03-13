import os

import matplotlib.pyplot as plt
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision
# import math
#
# from torchvision import transforms

# from torch.autograd import Variable

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchsummary import summary
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  # num_classes
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)  # VGG16 model
        net.classifier = nn.Sequential()  # 分類層放空
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),  # 512 * 7 * 7由VGG16決定  #((512 * 7 * 7, 512))
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),  # (512,128)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':

    # ##出現提示欄進行授權
    #
    # os.chdir('/content/drive/My Drive/DL')  # 切換該目錄
    # os.listdir()  # 確認目錄內容
    run()
    batch_size = 300
    LR = 0.001  # learning rate
    num_epoches = 40

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VGGNet()  # 創建model
    summary(model, input_size=(3, 32, 32), device='cpu')  # 印出網路層
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    model = model.to(device)
    # define loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    loss_p = np.array([])
    accuracy_train = np.array([])
    accuracy_test = np.array([])
    e = np.linspace(0, num_epoches, num_epoches)
    for epoch in range(num_epoches):
        model.train()
        print('\n', '*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        num_correct = 0.0
        for i, data in enumerate(trainloader, 0):
            img, label = data
            img, label = img.to(device), label.to(device)

            out = model(img)  # 往前

            # 向後
            loss = loss_func(out, label)  # 計算loss
            optimizer.zero_grad()  # 清空上一步殘餘的數值
            loss.backward()  # loss 求導，誤差反向傳播，計算參數更新直
            optimizer.step() # 更新參數:將參數更新值施加到net的parameters

            # 計算loss and correct
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)  # 預測最大值所在的label
            num_correct += (pred == label).sum().item()  # 計算正確個數

        print('Train==> Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(trainset)),
                                                                         num_correct / (len(trainset))))
        accuracy_train = np.append(accuracy_train, num_correct / (len(trainset)))

        # testing
        model.eval()
        eval_loss = 0
        num_correct = 0
        for data in testloader:
            img, label = data
            img, label = img.to(device).detach(), label.to(device).detach()

            out = model(img)
            loss = loss_func(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct += (pred == label).sum().item()
        print('Test==>  Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(testset)), num_correct / (len(testset))))
        loss_p = np.append(loss_p, running_loss / (len(trainset)))

        accuracy_test = np.append(accuracy_test, num_correct / (len(testset)))

    torch.save(model.state_dict(), './VGG16_2.pth')

    print('Finished training.')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(e, accuracy_train, color='blue', label='Training')
    plt.plot(e, accuracy_test, color='orange', label='Testing')
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(e, loss_p, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("./answer2.png")