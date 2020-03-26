import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import writer, yolo_utils
from dataset import NumberPlateDataset
from net import Net


class Detect():
    def __init__(self,
                 batch_size, lr0, epochs):
        self.batch_size = batch_size
        self.lr = lr0
        self.epochs = epochs
        self.device = ('cuda' if torch.cuda.is_available() else "cpu")
    
    def save_model(self, net, PATH):
        torch.save(net.state_dict(), PATH)

    def reload_model(self, PATH):
        net = Net()
        net.load_state_dict(torch.load(PATH))
        return net


    def train(self, trainloader):
        num_samples = len(trainloader.dataset)
        num_batches = math.ceil(num_samples/self.batch_size)
        ### initialize neural net
        net = Net()

        net.to(self.device)

        ### clear gradient buffer
        net.zero_grad()

        ### define optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                                     lr = self.lr*(10**(epochs/10)), weight_decay=5e-4)

        ### start training

        print("[#] Training started ...")
        epoch_loss = []
        for epoch in range(epochs):
            print("Epoch : %3d"%(epoch+1))
            running_loss = 0
            for data in trainloader:
                ### get inputs and annotations from image and xml file
                inputs = data['image'].to(self.device)
                true_boxes = data['true_boxes'].to(self.device)
                ### clear the gradient buffer of optimizer
                optimizer.zero_grad()

                ### forward propagation
                outputs = net(inputs)

                ### calculate loss
                loss = yolo_utils.Loss(batch_output = outputs, batch_true=true_boxes)

                ### backpropagate cost function
                loss.backward()

                ### update the weights
                optimizer.step()

                running_loss += loss.item()

                del outputs
                del inputs
                del true_boxes
                # print(f"Running loss : {round(running_loss, 5)}", end = "\r")
            running_loss /= num_batches
            epoch_loss.append(running_loss)
            # writer.log(epoch+1, running_loss, 0)
            # print(100*"=")
            # print("Epoch : %3d\tLoss : %.5f"%(epoch+1, running_loss))
        
        ### plot the loss during training
        plt.plot([i for i in range(1, self.epochs+1)], epoch_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss : Learning Rate = {self.lr}")
        plt.show()
        print("Training Complete :)")
        return net



def dataloader(images_path, labels_path, batch_size):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
    anchors = [(0.65, 0.31), (0.12,0.38), (0.56, 0.39), (0.7,0.3),(0.6,0.42)]
    train = NumberPlateDataset(images_path = images_path,
                               labels_path = labels_path,
                               anchors = anchors,
                               transform = transform)
    train_loader = DataLoader(train,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = 1)
    return train_loader


if __name__ == "__main__":

    ### parameters
    images_path = "../traffic_management/data/samples/images/"
    labels_path = "../traffic_management/data/samples/labels/"
    batch_size = 4
    lr0 = 0.00001
    epochs = 10

    train_loader = dataloader(images_path = images_path,
                              labels_path = labels_path,
                              batch_size = batch_size)
    d = Detect(lr0 = lr0,
               epochs = epochs,
               batch_size = batch_size)
    net = d.train(train_loader)
    os.makedirs("./checkpoints/", exist_ok = True)
    d.save_model(net, PATH = './checkpoints/saved_model.pth')
