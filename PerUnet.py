import torch.nn as nn
import torch

class PerUnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size= 3, stride= 1, padding= 1)
        #(16, 32, 32)
        self.bnd1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size= 3, stride= 1, padding= 1)
        #(16, 16, 16)
        self.bnd2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size= 3, stride= 1, padding= 1)
        # (16, 8, 8)
        self.bnd3 = nn.BatchNorm2d(16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # （16, 4, 4)
        self.bnd4 = nn.BatchNorm2d(16)

        self.conv11 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        # (4, 32, 32)
        self.bnd11 = nn.BatchNorm2d(4)

        self.conv22 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        # (4, 16, 16)
        self.bnd22 = nn.BatchNorm2d(4)
        self.upsample22 = nn.Upsample(scale_factor=2,mode='nearest')

        self.conv33 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        # (4, 8, 8)
        self.bnd33 = nn.BatchNorm2d(4)
        self.upsample33 = nn.Upsample(scale_factor=4, mode='nearest')

        self.conv44 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        # (4, 4, 4)
        self.bnd44 = nn.BatchNorm2d(4)
        self.upsample44 = nn.Upsample(scale_factor=8, mode='nearest')

        #self.conv5 = nn.Conv3d(1,1,(16, 8, 8),stride=1,padding=0)
        self.conv5 = nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1)

        self.linear = nn.Linear(1024, 20)

        #self.deconv = nn.ConvTranspose2d(1,1,kernel_size=2, stride=2)

        #self.final = nn.Linear(625, 20)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.bnd1(self.relu(self.conv1(x)))
        a2 = self.bnd2(self.relu(self.conv2(self.maxpool1(a1))))
        a3 = self.bnd3(self.relu(self.conv3(self.maxpool1(a2))))
        a4 = self.bnd4(self.relu(self.conv4(self.maxpool1(a3))))

        a11 = self.bnd11(self.relu(self.conv11(a1)))
        a22 = self.upsample22(self.bnd22(self.relu(self.conv22(a2))))
        a33 = self.upsample33(self.bnd33(self.relu(self.conv33(a3))))
        a44 = self.upsample44(self.bnd44(self.relu(self.conv44(a4))))

        a5 = torch.cat([a11,a22,a33,a44], dim=1)
        #a5 = a5.unsqueeze(dim=1)

        a6 = torch.reshape((self.conv5(a5)),(a5.shape[0],1024))
        #a7 = self.relu(self.linear(a6))
        a8 = self.sigmoid(self.linear(a6))

        #a7 = torch.reshape(self.relu(self.linear(a6)),(a5.shape[0], 25, 25))
        #a8 = self.sigmoid(self.deconv(a7.unsqueeze(dim=1))).squeeze(1)


        return a8



