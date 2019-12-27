import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
        #                        stride=1, padding=0, dilation=1, 
        #                        groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (7,7), stride = 2)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = 2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size = (3,3))
        self.norm2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride = 2)
        self.conv3_a = nn.Conv2d(192, 128, kernel_size=(1,1))
        self.norm3_a = nn.BatchNorm2d(128)
        self.conv3_b = nn.Conv2d(128, 256, kernel_size=(3,3))
        self.norm3_b = nn.BatchNorm2d(256)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size=(1,1))
        self.norm3_c = nn.BatchNorm2d(256)
        self.conv3_d = nn.Conv2d(256, 512, kernel_size=(3,3))
        self.norm3_d = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride = 2)
        self.conv4_a = nn.Conv2d(512, 256, kernel_size=(1,1))
        self.norm4_a = nn.BatchNorm2d(256)
        self.conv4_b = nn.Conv2d(256, 512, kernel_size=(3,3))
        self.norm4_b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride = 2)
        self.fc1 = nn.Linear(12*16*512, 12*16*64)
        self.norm_fc = nn.BatchNorm1d(12*16*64)
        self.conv5 = nn.Conv2d(64, 30, kernel_size= (1,1))
    
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
        return x