import torch.nn as nn
import torch

class MultiCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv13 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        #(8, 32, 32)
        self.conv15 = nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2)
        #(24, 16, 16)
        self.conv17 = nn.Conv2d(1, 48, kernel_size=7, stride=1, padding=3)
        #(48, 16, 16)
        self.conv19 = nn.Conv2d(1, 80, kernel_size=9, stride=1, padding=4)
        # (80, 16, 16)

        self.conv33 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        # (8, 32, 32)
        self.conv55 = nn.Conv2d(24, 8, kernel_size=5, stride=1, padding=2)
        # (24, 32, 32)
        self.conv77 = nn.Conv2d(48, 8, kernel_size=7, stride=1, padding=3)
        # (48, 32, 32)
        self.conv99 = nn.Conv2d(80, 8, kernel_size=9, stride=1, padding=4)
        # (8, 16, 16)

        self.conv32 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # (16, 8, 8)

        #self.conv81 = nn.Conv2d(21, 1, kernel_size=3, stride=1, padding=1)


        #self.conv55 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1)
        #self.conv77 = nn.Conv2d(64, 16, kernel_size=7, stride=1, padding=1)
        #self.conv99 = nn.Conv2d(128, 16, kernel_size=7, stride=1, padding=1)

        #self.conv31 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        #self.conv51 = nn.Conv2d(16, 4, kernel_size=5, stride=1, padding=2)
        #self.conv71 = nn.Conv2d(16, 4, kernel_size=7, stride=1, padding=3)

        self.bnd8 = nn.BatchNorm2d(8)
        self.bnd16 = nn.BatchNorm2d(16)
        self.bnd20 = nn.BatchNorm2d(20)
        self.bnd24 = nn.BatchNorm2d(24)
        self.bnd48 = nn.BatchNorm2d(48)
        self.bnd80 = nn.BatchNorm2d(80)
        self.bnd4 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        #self.upsample2 = nn.Upsample(scale_factor=2,mode='nearest')

        #self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        #self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')

        #self.conv3d = nn.Conv3d(1,1,(28, 8, 8),stride=1,padding=0)

        self.linear = nn.Linear(1024, 20)

        #self.deconv = nn.ConvTranspose2d(1,1,kernel_size=2, stride=2)

        #self.final = nn.Linear(625, 20)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        a3 = self.bnd8(self.maxpool(self.relu(self.conv33(self.bnd8(self.maxpool(self.relu(self.conv13(x))))))))
        a5 = self.bnd8(self.maxpool(self.relu(self.conv55(self.bnd24(self.maxpool(self.relu(self.conv15(x))))))))
        a7 = self.bnd8(self.maxpool(self.relu(self.conv77(self.bnd48(self.maxpool(self.relu(self.conv17(x))))))))
        a9 = self.bnd8(self.maxpool(self.relu(self.conv99(self.bnd80(self.maxpool(self.relu(self.conv19(x))))))))

        a3579 = torch.cat([a3, a5, a7, a9], dim=1)

        a = self.bnd16(self.relu(self.conv32(a3579)))
        # (16, 8, 8)

        #a = a.squeeze(dim=1)

        a = torch.reshape(a,(a.shape[0],1024))
        a = self.sigmoid(self.linear(a))

        #a7 = torch.reshape(self.relu(self.linear(a6)),(a5.shape[0], 25, 25))
        #a8 = self.sigmoid(self.deconv(a7.unsqueeze(dim=1))).squeeze(1)

        return a



