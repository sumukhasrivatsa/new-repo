import netA1
import netA2
import netA3
from torchvision import models
import torch.nn as nn

def get_neta_model(name,run):
    if name=="1":
        print("model 1")
        return netA1.netA1(run)
    elif name=="2":
        print("model 2")
        return netA2.netA2(run)


    elif name=="3":
        print("model 3")
        return netA3.netA3(run)

    else:
        return