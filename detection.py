import pandas as pd
import torch.nn as nn
from PIL import Image
import torch, torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math, os, sys, numpy as np
from torchvision import transforms
from YOLOUtils import *
from torch.utils.data import DataLoader, Dataset



class NumberPlateDataset(Dataset):

    def __init__(self, annotations_path,
                 transform = transforms.Compose([transforms.ToTensor()])):
        self.annotations_path = annotations_path
        self.transform = transform
        self.class_dict = {"license plate":1}
    
    def files(self):
        f = []
        for filename in os.listdir(self.annotations_path):
            f.append(filename)
        return f

    def __len__(self):
        return len(self.files())
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        xml_path = self.files()[idx]
        normalized_bb = NormalizedBB(os.path.join(self.annotations_path, xml_path))
        img = Image.open(normalized_bb['path'][0])

        if self.transform:
            img = self.transform(img)
        return {'image':img, 'bounding_boxes':normalized_bb}



class Net(nn.Module):

    # input image size (640, 480, 3)
    # Sx = 16
    # Sy = 12
    # B = Bounding Boxes = 5
    # C = Number of Classes = 1
    # Last layer : SXSX(B*5 + C) = 12X16X6

    def __init__(self):
        super(Net, self).__init__()

        #conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
        #                        stride=1, padding=0, dilation=1, 
        #                        groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (6,8), stride = 2) # (238, 317, 64)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # (119, 158, 64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size = (3,4)) # (117, 155, 192)
        self.norm2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # (58, 77, 192)
        self.conv3_a = nn.Conv2d(192, 128, kernel_size=(1,1)) # (58, 77, 128)
        self.norm3_a = nn.BatchNorm2d(128)
        self.conv3_b = nn.Conv2d(128, 256, kernel_size=(3,4)) # (56, 74, 256)
        self.norm3_b = nn.BatchNorm2d(256)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size=(1,1)) # (56, 74, 256)
        self.norm3_c = nn.BatchNorm2d(256)
        self.conv3_d = nn.Conv2d(256, 512, kernel_size=(3,4)) # (54, 71, 512)
        self.norm3_d = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # (27, 35, 512)
        self.conv4_a = nn.Conv2d(512, 256, kernel_size=(1,1)) # (27, 35, 256)
        self.norm4_a = nn.BatchNorm2d(256)
        self.conv4_b = nn.Conv2d(256, 512, kernel_size=(3,4)) # (25, 32, 512)
        self.norm4_b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # (12, 16, 512)
        self.fc1 = nn.Linear(12*16*512, 12*16*64) # (12*16*64)
        self.norm_fc = nn.BatchNorm1d(12*16*64)
        self.conv5 = nn.Conv2d(64, 30, kernel_size= (1,1)) # (12, 16, 30)
    
    def forward(self, x):
        x = self.pool1(self.norm1(F.leaky_relu(self.conv1(x))))
        x = self.pool2(self.norm2(F.leaky_relu(self.conv2(x))))
        x = self.norm3_a(F.leaky_relu(self.conv3_a(x)))
        x = self.norm3_b(F.leaky_relu(self.conv3_b(x)))
        x = self.norm3_c(F.leaky_relu(self.conv3_c(x)))
        x = self.pool3(self.conv3_d(F.leaky_relu(self.conv3_d(x))))
        x = self.norm4_a(F.leaky_relu(self.conv4_a(x)))
        x = self.pool4(self.norm4_b(F.leaky_relu(self.conv4_b(x))))
        x = self.norm_fc(F.relu(self.fc1(x)))
        x = x.view(12, 16, 64)
        x = self.conv5(x)


class Detect():
    def __init__(self, trainloader, 
                 lr0 = 0.001, epochs = 10, batch_size = 64):
        """
        Our  learning  rate  schedule  is  as  follows:  
        For  the  first epochs we slowly raise the learning rate from
        10−3 to 10−2. If we start at a high learning rate our model 
        often divergesdue to unstable gradients.  
        We continue training with 10−2 for 75 epochs, 
        then 10−3 for 30 epochs, and finally 10−4 for 30 epochs.
        """
        self.trainloader = trainloader
        self.lr = lr0*(10**(epochs/10))
        self.epochs = epochs
        self.batch_size = batch_size


    def train(self, number_of_samples):
        ### initialize neural net
        net = Net()

        ### clear gradient buffer
        net.zero_grad()

        ### define optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr = self.lr, momentum = 0.9)

        ### start training
        num_of_batches = math.ceil(number_of_samples/self.batch_size)

        print("[#] Training started...")
        epoch_loss = []
        for epoch in range(epochs):
            running_loss = 0
            for data in self.trainloader:
                ### get inputs and annotations from image and xml file
                inputs, annotations = data['image'], data['bounding_boxes']
                

                ### extract set of (x,y) and (w,h) from annotations
                print(annotations)

                ### clear the gradient buffer of optimizer
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.cost()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                print(f"Running loss : {round(running_loss, 5)}", end = "\r")
            running_loss /= num_of_batches
            epoch_loss.append(running_loss)
            print(100*"=")
            print(f"Epoch : {epoch + 1}")
            print(f"Loss : {round(running_loss,5)}")
        
        ### plot the loss during training
        plt.plot([i for i in range(1, self.epochs+1)], epoch_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss : Learning Rate = {self.lr}")
        plt.show()
        print("Training Complete :)")



def dataloader(train_path, batch_size = 2):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor()])
    train = NumberPlateDataset(annotations_path = train_path,
                               transform = transform)
    num_of_train_samples = len(train)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = 2)
    return train_loader, num_of_train_samples
    



if __name__ == "__main__":

    ### parameters
    path = "/media/mrityunjay/ExpandableDrive1/EDUCATIONAL/CSE/AI/Datasets/number_plate_dataset/train_annotations/"
    batch_size = 64
    lr = 0.01
    epochs = 10

    train_loader, num_of_train_samples = dataloader(train_path = path,
                                                    batch_size = batch_size)
    print(train_loader, num_of_train_samples)
    for data in train_loader:
        image, annotation = data['image'], data['bounding_boxes']
        print(annotation['x'])
        print(annotation['y'])