import pandas as pd
import torch.nn as nn
from PIL import Image
import torch, torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math, os, sys, numpy as np
from torchvision import transforms
from net import Net
from dataset import NumberPlateDataset
from torch.utils.data import DataLoader, Dataset



class Detect():
    def __init__(self, trainloader, 
                 lr0 = 0.001, epochs = 10, batch_size = 64):
        """
        Our  learning  rate  schedule  is  as  follows:  
        For  the  first epochs we slowly raise the learning rate from
        10−3 to 10−2. If we start at a high learning rate our model 
        often diverges due to unstable gradients.  
        We continue training with 10−2 for 75 epochs, 
        then 10−3 for 30 epochs, and finally 10−4 for 30 epochs.
        """
        self.trainloader = trainloader
        self.lr = lr0
        self.epochs = epochs
        self.batch_size = batch_size
    
    def save_model(self, net, PATH):
        torch.save(net.state_dict(), PATH)

    def reload_model(self, PATH):
        net = Net()
        net.load_state_dict(torch.load(PATH))
        return net


    def train(self, number_of_samples):
        ### initialize neural net
        net = Net()

        ### clear gradient buffer
        net.zero_grad()

        ### define optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                                     lr = self.lr*(10**(epochs/10)), weight_decay=5e-4)

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

                ### clear the gradient buffer of optimizer
                optimizer.zero_grad()
                outputs = net(inputs)
                # loss = cost()
                # loss.backward()
                optimizer.step()

                # running_loss += loss.item()
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
        return net



def dataloader(train_images_path, train_annotations_path, batch_size = 64):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
    anchors = [(0.65, 0.31), (0.12,0.38), (0.56, 0.39), (0.7,0.3),(0.6,0.42)]
    train = NumberPlateDataset(train_images_path = train_images_path,
                               anchors = anchors,
                               annotations_path = train_annotations_path,
                               transform = transform)
    num_of_train_samples = len(train)
    train_loader = DataLoader(train,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = 2)
    return train_loader, num_of_train_samples


if __name__ == "__main__":

    ### parameters
    train_path = "/media/mrityunjay/ExpandableDrive/EDUCATIONAL/CSE/AI/Datasets/number_plate_dataset/train/"
    annotations_path = "/media/mrityunjay/ExpandableDrive/EDUCATIONAL/CSE/AI/Datasets/number_plate_dataset/train_annotations/"
    batch_size = 64
    lr = 0.01
    epochs = 10

    train_loader, num_of_train_samples = dataloader(train_images_path = train_path,
                                                    train_annotations_path = annotations_path,
                                                    batch_size = batch_size)
    for data in train_loader:
        batch_images, batch_true_boxes = data['image'], data['true_boxes']