import torch.nn as nn
import torch.optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import model_selector_netc
import model_selector_neta
import parameters_calculator
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter




class trainer_class(nn.Module):
    def __init__(self,train_loader,validation_loader,args,run):
        super().__init__()
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.run=run
        self.args=args


    def train(self):
        count=0
        ##do not write outer loop from here 
        
        #writer = SummaryWriter('/experiments-AUTOMATIC-DATA-AUGMENTATION')
        neta_obj=model_selector_neta.get_neta_model("1",self.run)
        n_parameters=parameters_calculator.count_parameters(neta_obj)
        print(n_parameters)
        opt_neta=torch.optim.Adam(neta_obj.fc_loc.parameters(),lr=0.005,weight_decay=1e-5)
        for param_group in opt_neta.param_groups:
            print(param_group['lr'])

        ##need to get netc_obj object here
        ##tell name of model to helper.get_netc_model()
        netc_obj=model_selector_netc.get_netc_model("custom")
        opt_netc=torch.optim.Adam(netc_obj.parameters(),lr=0.001)
        loss_obj=nn.CrossEntropyLoss()
        
       
        count_tr=0
        count2_tr=0
        count_v=0
        count2_v=0
        avger=0
        accuracy_t=0
        accuracy_v=0
        for epoch in range(30):
            val_loader=iter(self.validation_loader)
            tr_loader=iter(self.train_loader)
            
            for i in range(len(self.train_loader)):
                print("epoch {}".format(epoch))
                ##define outer loop from here 

                ###get images from train loader
                images,labels=next(tr_loader)

                count_tr=0
                count2_tr=0


                rot_imgs,rot_labels,affine_matrix,inverse_matrix=neta_obj.transform_images(images,labels)
                output1=netc_obj.forward(rot_imgs)
                loss1=loss_obj(output1,labels)
                opt_netc.zero_grad()
                opt_neta.zero_grad()

                loss1.backward(retain_graph=True)
                

                #print(neta_obj.fc_loc[0].weight.grad)
                ##UPDATE only netc_obj
                opt_netc.step()
                opt_netc.zero_grad()
                opt_neta.zero_grad()

                output=netc_obj.forward(rot_imgs)
                
                ##create an object of neta_obj and send image batch
                
                avger+=1
                ##defining an optimizer for neta_obj
                m=0
                for j in output:
                    x=torch.max(j)
                    
                    c=((j == x).nonzero(as_tuple=True)[0])
                    if labels[m]==c:
                        count_tr+=1
                    count2_tr+=1
                    m+=1
                
                accuracy_t=accuracy_t+(count_tr/count2_tr*100)
                accuracy_t_avg=accuracy_t/avger
                self.run['train/accuracy'].log(accuracy_t_avg)

                    #writer.add_scalar("accuracy",accuracy)
                ##^ obtained the rotated images from neta_obj
                ##now we have both images and rot_imgs
                
                ##object of netc_obj
                

                ##passing rotated images through netc_obj

                
                
                ##defining loss
                
                loss=loss_obj(output,labels)
                self.run['train/loss'].log(loss)
                ##gradients before backward for neta_obj
                #for param in neta_obj.fc_loc.parameters():
                #    print(param.data)

                ##gradients before backward for netc_obj
                #for param in netc_obj.parameters():
                #   print(param.data)
                
                #print(neta_obj.fc_loc[0].weight.grad)
                #print(netc_obj.conv1.weight.grad)

                opt_netc.zero_grad()
                opt_neta.zero_grad()

                loss.backward(retain_graph=True)
                

                #print(neta_obj.fc_loc[0].weight.grad)
                ##UPDATE only netc_obj
                opt_netc.step()

                opt_netc.zero_grad()
                opt_neta.zero_grad()
                



                ##print(netc_obj.conv1.weight.grad)


                
                ##get the next from the validation batch
                val_imgs,val_labels=val_loader.next()
                
                
                

                    
                    
                
                
                # apply transformation +ve
                grids1 = F.affine_grid(affine_matrix[0:len(val_imgs),:,:], val_imgs.size(), align_corners=True)
                rot_val_imgs = F.grid_sample(val_imgs,grids1, align_corners=True)

                
                    
                

                ##apply inverse transformation
                grids2 = F.affine_grid(inverse_matrix[0:len(val_imgs),:,:], val_imgs.size(), align_corners=True)
                rot_val_imgs_inv = F.grid_sample(rot_val_imgs,grids2, align_corners=True)
                """
                if i%400==0:
                    img1=val_imgs[3][0,:,:].numpy()
                    img2=rot_val_imgs[3][0,:,:].detach().numpy()
                    img3=rot_val_imgs_inv[3][0,:,:].detach().numpy()
                    
                    plt.figure()

                #subplot(r,c) provide the no. of rows and columns
                    f, axarr = plt.subplots(3,1) 

                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                    axarr[0].imshow(img1)
                    axarr[1].imshow(img2)
                    axarr[2].imshow(img3)
                    plt.show()
                """
                
                count_v=0
                count2_v=0           
                ##passing the untransformed(transformed) images through netc_obj
                outputs2=netc_obj.forward(rot_val_imgs_inv)
                
                y=0
                for i in outputs2:
                    x=torch.max(i)
                    
                    c=((i == x).nonzero(as_tuple=True)[0])
                    if val_labels[y]==c:
                        count_v+=1
                    count2_v+=1
                    y+=1
            
                accuracy_v=accuracy_v+(count_v/count2_v*100)
                accuracy_v_avg=accuracy_v/avger
                self.run['train/val_accuracy'].log(accuracy_v_avg)

                    
            
                

                
                    

                

                
                    
                val_loss=loss_obj(outputs2,val_labels)
                self.run['train/val_loss'].log(val_loss)
                ##zeroing the grad from previous steps 
                opt_neta.zero_grad()
                opt_netc.zero_grad()

                #backpropogating with validation loss 
                val_loss.backward(retain_graph=True)
                

                ##optimizing only neta_obj
                opt_neta.step()
            

                
                
                

                ##neta_obj gradients after backward
                #for param in neta_obj.fc_loc.parameters():
                #    print(param.data)
                
                ##netc_obj gradients after backward

                #for param in netc_obj.parameters():
                #   print(param.data)
                
                #rots_images=rot_imgs[5,0,:,:]
                #rots_images=rots_images.detach().numpy()
                
                #rots_images=rot_imgs[0,0,:,:]
                #rots_images=rots_images.detach().numpy()
                #print(rots_images.shape)
                """
                print("accuracy={}   val_loss== {}  epoch=={}".format((count/count2*100),(loss),epoch))"""

                
          
            

                ###thing is i need to use the same grid with same neta_obj 
                
            
        return netc_obj        
            
            ###pass them to get the noise and the transformed images
            ## pass the transformed and regular images to the netc_obj
            ##use the loss to only back propogate through the netc_obj and not neta_obj