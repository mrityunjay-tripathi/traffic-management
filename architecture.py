import cv2
import math
import torch
import numpy as np
import torchvision
import torch.nn as nn
import os, sys, argparse
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot  as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader




class CustomDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_dict = {"airplane":torch.eye(8)[0],
                           "car":torch.eye(8)[1],
                           "cat":torch.eye(8)[2],
                           "dog":torch.eye(8)[3],
                           "flower":torch.eye(8)[4],
                           "fruit":torch.eye(8)[5],
                           "motorbike":torch.eye(8)[6],
                           "person":torch.eye(8)[7]}
    
    def __len__(self):
        return len(self.files())
    
    def files(self):
        f = []
        for filename in os.listdir(self.root_dir):
            f.append(filename)
        return f
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.files()[idx]
        img = Image.open(os.path.join(self.root_dir, img_name))
        label = self.class_dict[img_name[:-9]]

        if self.transform:
            img = self.transform(img)
        return {'image':img, 'label':label}

def data_loader(batch_size = 64):
    tf = transforms.Compose([transforms.Resize((224,224)),
                             transforms.ToTensor()])
    
    train = CustomDataset(root_dir = train_path, transform = tf)
    num_of_train_samples = len(train)
    train_loader = DataLoader(train, 
                              batch_size = batch_size, 
                              shuffle = True, 
                              num_workers=2)
    
    test = CustomDataset(root_dir = test_path, transform = tf)
    num_of_test_samples = len(test)
    test_loader = DataLoader(test,
                             batch_size = batch_size,
                             shuffle = True,
                             num_workers = 2)
    return train_loader, test_loader, num_of_train_samples, num_of_test_samples



class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()
        
        #conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
        #                        stride=1, padding=0, dilation=1, 
        #                        groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 64, kernel_size = (5,5), stride = 2)#(222,222,64)
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2), stride=2)#(111,111,64)

        self.conv2 = nn.Conv2d(64, 192, kernel_size = (3,3))#(109,109,192)
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2), stride=2)#(54,54,192)

        self.conv3_a = nn.Conv2d(192, 128, kernel_size = (1,1))#(54,54,128)
        self.conv3_b = nn.Conv2d(128, 256, kernel_size = (3,3))#(52,52,256)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size = (1,1))#(52,52,256)
        self.conv3_d = nn.Conv2d(256, 512, kernel_size = (3,3))#(50,50,512)
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2), stride=2)#(25,25,512)

        self.conv4_a = nn.Conv2d(512, 256, kernel_size = (1,1))#(25,25,256)
        self.conv4_b = nn.Conv2d(256, 512, kernel_size = (3,3))#(23,23,512)
        self.conv4_c = nn.Conv2d(512, 256, kernel_size = (1,1))#(23,23,256)
        self.conv4_d = nn.Conv2d(256, 512, kernel_size = (3,3))#(21,21,512)
        self.conv4_e = nn.Conv2d(512, 256, kernel_size = (1,1))#(21,21,256)
        self.conv4_f = nn.Conv2d(256, 512, kernel_size = (3,3))#(19,19,512)
        self.conv4_g = nn.Conv2d(512, 256, kernel_size = (1,1))#(19,19,256)
        self.conv4_h = nn.Conv2d(256, 512, kernel_size = (3,3))#(17,17,512)
        self.conv4_i = nn.Conv2d(512, 1024, kernel_size = (1,1))
        self.pool4 = nn.AvgPool2d(kernel_size = (2,2), stride = 2)
        self.fc1 = nn.Linear(in_features = 1024, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 8)
    
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3_b(F.leaky_relu(self.conv3_a(x))))
        x = self.pool3(F.leaky_relu(self.conv3_d(F.leaky_relu(self.conv3_c(x)))))
        
        x = F.leaky_relu(self.conv4_b(F.leaky_relu(self.conv4_a(x))))
        x = F.leaky_relu(self.conv4_d(F.leaky_relu(self.conv4_c(x))))
        x = F.leaky_relu(self.conv4_f(F.leaky_relu(self.conv4_e(x))))
        x = F.leaky_relu(self.conv4_h(F.leaky_relu(self.conv4_g(x))))
        x = self.pool4(F.leaky_relu(self.conv4_i(x)))
        x = x.view(x.shape[0], x.shape[1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classify():
    def __init__(self, trainloader, 
                lr = 0.001, epochs = 10, batch_size = 64):
        self.trainloader = trainloader
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train_model(self, num_of_samples):
        ### initialize neural net 
        net = YOLO()

        ### clear gradient buffer
        net.zero_grad()

        ### define optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr = self.lr, momentum = 0.9)

        ### start training
        num_of_batches = math.ceil(num_of_samples/self.batch_size)

        print("[#]Training started...")
        epoch_loss = []
        for epoch in range(self.epochs):
            running_loss = 0
            for data in self.trainloader:
                ### get inputs and labels from dataloader
                inputs, labels = data['image'], data['label']

                ### clear the gradient buffer of optimizer
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss+=loss.item()
                print("Running Loss : {}".format(round(running_loss,5)), end = "\r")
            
            running_loss/=num_of_batches
            epoch_loss.append(running_loss)
            print(100*"=")
            print(f"Epoch : {epoch + 1}")
            print(f"Loss : {round(running_loss,5)}")
        
        ### plot loss during training
        plt.plot([i for i in range(1, self.epochs+1)], epoch_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss : Learning Rate = {self.lr}")
        plt.show()
        print("Training Complete :)")

        return net
    
    def predict(self, net, testloader, num_of_samples):
        ### test the model

        dataiter = iter(testloader)
        predicted = []
        accuracy = 0
        while(True):
            try:
                batch = dataiter.next()
                test_images, test_labels = batch['image'], batch['label']
                test_outputs = net(test_images)
                _, p = torch.max(test_outputs, 1)
                accuracy+=torch.sum(1*(torch.argmax(test_labels)==p), 
                                    dtype = torch.float32)
                predicted.append(p)
            except StopIteration:
                break
        
        accuracy = torch.div(accuracy, num_of_samples).item()
        return predicted, accuracy


### set parameters
train_path = "/media/mrityunjay/ExpandableDrive1/EDUCATIONAL/CSE/AI/Datasets/natural-images/train/"
test_path = "/media/mrityunjay/ExpandableDrive1/EDUCATIONAL/CSE/AI/Datasets/natural-images/test/"
batch_size = 64
lr = 0.1
epochs = 1

### get data and size of data
trainloader, testloader, train_len, test_len = data_loader(batch_size=64)


### define your Classify class
c = Classify(trainloader = trainloader, lr = lr,
             epochs=epochs, batch_size=batch_size)

### train your model
net = c.train_model(num_of_samples = train_len)

### save
torch.save(net.state_dict(), f="./saved_model")


### load saved model
net_loaded = YOLO()
net_loaded.load_state_dict(torch.load('./saved_model'))

### predict using loaded model
pred, acc = c.predict(net_loaded, testloader, test_len)
print("Accuracy :", acc)