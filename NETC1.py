import torch.nn.functional as F
from torch.distributions import Uniform
import torch.nn as nn
import torch

#add tensor board features
#add baseline model and compare the accuracies  
class netC(nn.Module):
    def __init__(self):
        super(netC, self).__init__()
        
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        #(10,3,350,350)
       
        #Input shape= (10,1,350,350)
        
        self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        #Shape= (32,4,28,28)
        
        #Shape= (32,4,28,28)
        self.relu1=nn.ReLU()

        
        #Shape= (32 4,28,28)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (32,4,14,14)
        
        
        self.conv2=nn.Conv2d(in_channels=4,out_channels=10,kernel_size=3,stride=1,padding=1)
        #Shape= (32,10,14 14)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        ## 32 10 7 7 
        self.conv3=nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (32,32 7 7 )
        
        #Shape= (32,32 7 7 
        self.relu3=nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=20,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.relu4=nn.ReLU()

        #Shape= (32,5,14 14)

        
        self.fc1=nn.Linear(in_features=24*7*7,out_features=300)
        self.fc2=nn.Linear(300,100)
        self.fc3=nn.Linear(100,10)
        self.out_=nn.LogSoftmax(dim=1)
    
    def forward(self,images):
        inputs=images
        output=self.conv1(inputs)
        #print(output.shape)
        
        output=self.relu1(output)
        #print(output.shape)
            
        output=self.pool(output)
        #print(output.shape)    
        output=self.conv2(output)
        #print(output.shape)
        output=self.relu2(output)
        #print(output.shape)
        output=self.pool2(output)
        output=self.conv3(output)
        
        output=self.relu3(output)
        output=self.conv4(output)
        
        output=self.relu4(output)
            
        
            
            
        #Above output will be in matrix form, with shape (10,2,25,25)
            
        output=output.view(-1,24*7*7)
            
            
        output=self.fc1(output)
        #print(output.shape)
        output=self.fc2(output)
        output=self.fc3(output)
        #print(output.shape)
        output=self.out_(output)
        #print(output.shape)
            
        return output


        