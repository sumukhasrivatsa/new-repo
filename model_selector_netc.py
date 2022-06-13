import NETC1
from torchvision import models
import torch.nn as nn
import VGG16
def get_netc_model(name):
    if name=="custom":
        return NETC1.netC()
    elif name=="resnet":
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048, 10, bias=True)
        return model


    elif name=="vgg16":
        model=VGG16.VGG16()
        return model