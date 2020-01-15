import torch.nn as nn
import torch.nn.functional as F

configurations = {}
with open("../traffic_management/config/params.cfg", "r+") as config:
    for line in config:
        key, value = line.split("=")
        if key:
            configurations[key] = eval(value)

grid_shape = configurations['grid_shape']
num_anchors = len(configurations['anchors'])
num_classes = configurations['num_classes']
del configurations


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
        #                        stride=1, padding=0, dilation=1, 
        #                        groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (7,7), stride = 2) # 237, 317
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # 118, 158

        self.conv2 = nn.Conv2d(64, 192, kernel_size = (3,3)) # 116, 156
        self.norm2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # 58, 78

        self.conv3_a = nn.Conv2d(192, 128, kernel_size=(1,1)) # 58, 78
        self.norm3_a = nn.BatchNorm2d(128)
        self.conv3_b = nn.Conv2d(128, 256, kernel_size=(3,3)) # 56, 76
        self.norm3_b = nn.BatchNorm2d(256)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size=(1,1)) # 56, 76
        self.norm3_c = nn.BatchNorm2d(256)
        self.conv3_d = nn.Conv2d(256, 512, kernel_size=(3,3)) # 54, 74
        self.norm3_d = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # 27, 37

        self.conv4_a = nn.Conv2d(512, 256, kernel_size=(1,1)) # 27, 37
        self.norm4_a = nn.BatchNorm2d(256)
        self.conv4_b = nn.Conv2d(256, 512, kernel_size=(3,3)) # 25, 35
        self.norm4_b = nn.BatchNorm2d(512)
        self.conv4_c = nn.Conv2d(512, 256, kernel_size=(1,1)) # 25, 35
        self.norm4_c = nn.BatchNorm2d(256)
        self.conv4_d = nn.Conv2d(256, 512, kernel_size=(3,3)) # 23, 33
        self.norm4_d = nn.BatchNorm2d(512)
        self.conv4_e = nn.Conv2d(512, 256, kernel_size=(1,1)) # 23, 33
        self.norm4_e = nn.BatchNorm2d(256)
        self.conv4_f = nn.Conv2d(256, 512, kernel_size=(3,3)) # 21, 31
        self.norm4_f = nn.BatchNorm2d(512)
        self.conv4_g = nn.Conv2d(512, 256, kernel_size=(1,1)) # 21, 31
        self.norm4_g = nn.BatchNorm2d(256)
        self.conv4_h = nn.Conv2d(256, 512, kernel_size=(3,3)) # 19, 29
        self.norm4_h = nn.BatchNorm2d(512)
        self.conv4_i = nn.Conv2d(512, 512, kernel_size=(1,1)) # 19, 29
        self.norm4_i = nn.BatchNorm2d(512)
        self.conv4_j = nn.Conv2d(512, 1024, kernel_size=(3,3)) # 17, 27
        self.norm4_j = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride = 2) # 8, 13

        self.conv5_a = nn.Conv2d(1024, 512, kernel_size=(1,1)) # 8, 13
        self.norm5_a = nn.BatchNorm2d(512)
        self.conv5_b = nn.Conv2d(512, 1024, kernel_size=(3,3)) # 6, 11
        self.norm5_b = nn.BatchNorm2d(1024)
        self.conv5_c = nn.Conv2d(1024, 512, kernel_size=(1,1)) # 6, 11
        self.norm5_c = nn.BatchNorm2d(512)
        self.conv5_d = nn.Conv2d(512, 1024, kernel_size=(3,3)) # 4, 9
        self.norm5_d = nn.BatchNorm2d(1024)
        self.conv5_e = nn.Conv2d(1024, 1024, kernel_size=(3,3)) # 2, 7
        self.norm5_e = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(kernel_size = (2,2), stride = 2) # 1, 3

        in_features = 1*3*1024
        out_features = grid_shape[0]*grid_shape[1]*num_anchors*(5 + num_classes)

        self.fc1 = nn.Linear(in_features, out_features)
        self.conv5 = nn.Conv2d(num_anchors*(5+num_classes), 
                               num_anchors*(5+num_classes), kernel_size= (1,1))
    
    def forward(self, x):
        x = self.pool1(self.norm1(F.leaky_relu(self.conv1(x))))
        x = self.pool2(self.norm2(F.leaky_relu(self.conv2(x))))

        x = self.norm3_a(F.leaky_relu(self.conv3_a(x)))
        x = self.norm3_b(F.leaky_relu(self.conv3_b(x)))
        x = self.norm3_c(F.leaky_relu(self.conv3_c(x)))
        x = self.norm3_d(F.leaky_relu(self.conv3_d(x)))
        x = self.pool3(x)

        x = self.norm4_a(F.leaky_relu(self.conv4_a(x)))
        x = self.norm4_b(F.leaky_relu(self.conv4_b(x)))
        x = self.norm4_c(F.leaky_relu(self.conv4_c(x)))
        x = self.norm4_d(F.leaky_relu(self.conv4_d(x)))
        x = self.norm4_e(F.leaky_relu(self.conv4_e(x)))
        x = self.norm4_f(F.leaky_relu(self.conv4_f(x)))
        x = self.norm4_g(F.leaky_relu(self.conv4_g(x)))
        x = self.norm4_h(F.leaky_relu(self.conv4_h(x)))
        x = self.norm4_i(F.leaky_relu(self.conv4_i(x)))
        x = self.norm4_j(F.leaky_relu(self.conv4_j(x)))
        x = self.pool4(x)

        x = self.norm5_a(F.leaky_relu(self.conv5_a(x)))
        x = self.norm5_b(F.leaky_relu(self.conv5_b(x)))
        x = self.norm5_c(F.leaky_relu(self.conv5_c(x)))
        x = self.norm5_d(F.leaky_relu(self.conv5_d(x)))
        x = self.norm5_e(F.leaky_relu(self.conv5_e(x)))
        x = self.pool5(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        new_shape = (x.shape[0], num_anchors*(5 + num_classes), grid_shape[0], grid_shape[1])
        x = x.view(new_shape)
        x = self.conv5(x)
        return x