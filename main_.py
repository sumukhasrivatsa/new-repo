import torch
import argparse
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.distributions import Uniform
import glob
from torchvision import datasets
from torch.utils.data import DataLoader 
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision.transforms import transforms
torch.manual_seed(100)
import neptune.new as neptune
import matplotlib.pyplot as plt

print("no errors")
import trainer1,trainer_



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_norm", default = True, type = bool)
    parser.add_argument("--batch_size_train", default = 30, type = int)
    parser.add_argument("--batch_size_val", default = 20, type = int)
    parser.add_argument("--criterion_type", default = "cross-entropy", type = str)
    parser.add_argument("--data_path", default = "./datasets/MNIST/", type = str)
    parser.add_argument("--download_data", default = False, type = bool)
    parser.add_argument("--init_weights", default = True, type = bool)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--num_classes", default = 10, type = int)
    parser.add_argument("--num_epochs", default = 100, type = int)
    parser.add_argument("--num_val_examples", default = 1000, type = int)
    parser.add_argument("--optimizer_type", default = "Adam", type = str)
    parser.add_argument("--progress", default = True, type = bool)
    parser.add_argument("--seed", default = 0, type = int)

    run = neptune.init(
    project="sumukha3011/ADA-Automatic-Data-Augmentation",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NTM0NzU2Mi0xMTQ3LTRjNmItYTk0Mi1mYmFlMTBhZDU5NTcifQ==",
    )

    args = parser.parse_args()
    path_train=r'C:\Users\asus\OneDrive\Desktop\ADA\train_set'
    path_val=r'C:\Users\asus\OneDrive\Desktop\ADA\val_set'
    path_test=r'C:\Users\asus\OneDrive\Desktop\ADA\test_set'
    print("data set done")

  


    #defining transformations 1 and 2
    Transform=transforms.Compose([
    
    transforms.ToTensor()
    
    ])
    Transform2=transforms.Compose([
    transforms.RandomRotation(15),
    
    transforms.ToTensor()
    
    ])
    Transform3=transforms.Compose([
    transforms.RandomRotation(65),
    
    transforms.ToTensor()
    
    ])

    

    
    ##training set with download set to true
    train_set = datasets.MNIST(path_train, download=True, train=True, transform=Transform)
    val_set=datasets.MNIST(path_train, download=True, train=True, transform=Transform)
    train_set2 = datasets.MNIST(path_train, download=True, train=True, transform=Transform2)
    train_set1=list(train_set)[0:30000]
    print(len(train_set1))
    #testing set with download set to true
    val_set = list(val_set)[30000:50000]

    
    

    test_set=datasets.MNIST(path_test,download=True, train=False, transform=Transform3)

    #train loader with batch size 32 which gives us batches
    train_loader = torch.utils.data.DataLoader(train_set1, batch_size=args.batch_size_train, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=args.batch_size_train, shuffle=True)
    print(len(train_loader))
    #val loader with batch size 32 which gives us batches of 32
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_val, shuffle=True)
    print(len(val_loader))
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)
    print(len(test_loader))
   

    mean=torch.tensor(0)
    std=torch.tensor(0)
    nz=6
    print(len(train_set),len(train_set),len(train_set))
    print("*********************************************************************")
    trainer_class="trainer1"
    trainer=None
    if trainer_class=="trainer1":

        trainer=trainer1.trainer_class(train_loader,val_loader,args,run)
    else:
        trainer=trainer_.trainer_class(train_loader2,val_loader,args,run)
    model=trainer.train()
    test_loss_object=nn.CrossEntropyLoss()
    
    avger=0
    accuracy=0
    for i,(image,labels) in enumerate(test_loader):
        if i%100==0:
            img1=image[0][0,:,:].detach().numpy()
            plt.imshow(img1)
            plt.show()
        avger+=1
        counter=counter2=0
        with torch.no_grad():
            outputs=model.forward(image)
        m=0
        for i in outputs:
            x=torch.max(i)
            c=((i == x).nonzero(as_tuple=True)[0])
            if labels[m]==c:
                counter+=1
            counter2+=1
            m+=1
        accuracy=accuracy+(counter/counter2*100)
        run["testing/accuracy"].log(accuracy/avger)
        print("test_accuracy===={}".format(accuracy/avger))
    ##defining the tester
     
    

    
    

    






            


