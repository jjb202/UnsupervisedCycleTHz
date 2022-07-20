""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.upc1 = UpTwo(128, 128, bilinear)
        self.upc2 = UpTrr(64, 32, bilinear)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, 2),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, 4),
            nn.ReLU(inplace=True),
        )
        #########################
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, 8),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, 16),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.DeConv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.DeConv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.DeConv1(x12)
        x14 = self.upc1(x13, x4)
        x15 = self.DeConv2(x14)
        x16 = self.upc2(x15, x2)
        x17 = self.conv15(x16)

        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        return x17
