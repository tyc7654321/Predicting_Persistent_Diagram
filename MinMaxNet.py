import torch.nn as nn
import torch


class MinMaxNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        #self.conv33 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        #self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        #self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        # self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')

        # self.conv3d = nn.Conv3d(1,1,(28, 8, 8),stride=1,padding=0)

        self.linear = nn.Linear(512, 40)

        # self.deconv = nn.ConvTranspose2d(1,1,kernel_size=2, stride=2)

        self.final = nn.Linear(40, 20)

        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        a1 = self.maxpool(x)
        a2 = -1 * self.maxpool(-1 * x)

        a11 = self.maxpool(a1)
        a12 = -1 * self.maxpool(-1 * self.conv2(a1))
        a21 = self.maxpool(self.conv3(a2))
        a22 = -1 * self.maxpool(-1 * a2)

        a3 = torch.cat([a11, a12, a21, a22], dim=1)

        a5 = self.relu(self.conv1(a3))

        a6 = torch.reshape(a5, (a5.shape[0], 512))

        a7 = self.relu(self.linear(a6))

        a7 = self.leakyrelu(self.final(a7))


        return a7



