import torch
import torch.nn as nn


def rgb2y(img):
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.2126*r + 0.7152*g + 0.0722*b
    return y


def linenormalize(imgo):
    img = torch.reshape(imgo, (imgo.size()[0], 1, -1))
    ymax = torch.max(img, 2, keepdim=True)[0].unsqueeze(-1)
    ymin = torch.min(img, 2, keepdim=True)[0].unsqueeze(-1)
    y = (imgo - ymin)/(ymax - ymin)
    return y


class Dsc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dsc, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.reflect = nn.ReflectionPad2d(1)

    def forward(self, xin):
        xin = self.reflect(xin)
        xout = self.depth_conv(xin)
        xout = self.point_conv(xout)
        return xout


class Ctmo(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Ctmo, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1
        )
        self.reflect = nn.ReflectionPad2d(1)

    def forward(self, xin):
        xin = self.reflect(xin)
        xout = self.conv(xin)
        return xout


class Tmonet(nn.Module):
    def __init__(self, scale_factor):
        super(Tmonet, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        # self.sigmoid = nn.ReLU(inplace=True)
        self.sigmoid = torch.sigmoid
        self.scale_factor = scale_factor
        number_f = 16

        self.e_conv11 = Ctmo(1, number_f)
        self.e_conv21 = Ctmo(number_f, number_f)
        self.e_conv31 = Ctmo(number_f, number_f)
        # self.e_conv11 = Dsc(3, number_f)
        # self.e_conv21 = Dsc(number_f, number_f)
        # self.e_conv31 = Dsc(number_f, number_f)
        self.e_conv41 = Dsc(number_f, number_f)
        self.e_conv51 = Dsc(number_f*2, number_f)
        self.e_conv61 = Dsc(number_f*2, number_f)
        self.e_conv71 = Dsc(number_f*2, 1)

        # self.e_conv12 = Dsc(3, number_f)
        # self.e_conv22 = Dsc(number_f, number_f)
        # self.e_conv32 = Dsc(number_f, number_f)
        self.e_conv42 = Dsc(number_f, number_f)
        self.e_conv52 = Dsc(number_f*2, number_f)
        self.e_conv62 = Dsc(number_f*2, number_f)
        self.e_conv72 = Dsc(number_f*2, 1)

        # self.e_conv13 = Dsc(3, number_f)
        # self.e_conv23 = Dsc(number_f, number_f)
        # self.e_conv33 = Dsc(number_f, number_f)
        self.e_conv43 = Dsc(number_f, number_f)
        self.e_conv53 = Dsc(number_f*2, number_f)
        self.e_conv63 = Dsc(number_f*2, number_f)
        self.e_conv73 = Dsc(number_f*2, 1)

        # self.e_conv14 = Dsc(3, number_f)
        # self.e_conv24 = Dsc(number_f, number_f)
        # self.e_conv34 = Dsc(number_f, number_f)
        self.e_conv44 = Dsc(number_f, number_f)
        self.e_conv54 = Dsc(number_f*2, number_f)
        self.e_conv64 = Dsc(number_f*2, number_f)
        self.e_conv74 = Dsc(number_f*2, 1)

    def forward(self, x):
        lum = rgb2y(x)
        x10 = torch.log(lum + 0.00001)
        if self.scale_factor == 1:
            x_down = x10
        else:
            x_down = nn.functional.interpolate(x10, scale_factor=1 / self.scale_factor, mode='bilinear', align_corners=True)
        x11 = self.relu(self.e_conv11(x_down))
        x21 = self.relu(self.e_conv21(x11))
        x31 = self.relu(self.e_conv31(x21))
        # x41 = self.relu(self.e_conv41(x31))
        # x51 = self.relu(self.e_conv51(torch.cat([x31, x41], 1)))
        # x61 = self.relu(self.e_conv61(torch.cat([x21, x51], 1)))
        # r1 = self.sigmoid(self.e_conv71(torch.cat([x11, x61], 1)))

        # x12 = self.relu(self.e_conv12(x_down))
        # x22 = self.relu(self.e_conv22(x12))
        # x32 = self.relu(self.e_conv32(x22))
        x42 = self.relu(self.e_conv42(x31))
        x52 = self.relu(self.e_conv52(torch.cat([x31, x42], 1)))
        x62 = self.relu(self.e_conv62(torch.cat([x21, x52], 1)))
        r2 = self.sigmoid(self.e_conv72(torch.cat([x11, x62], 1)))

        # x13 = self.relu(self.e_conv13(x_down))
        # x23 = self.relu(self.e_conv23(x13))
        # x33 = self.relu(self.e_conv33(x23))
        x43 = self.relu(self.e_conv43(x31))
        x53 = self.relu(self.e_conv53(torch.cat([x31, x43], 1)))
        x63 = self.relu(self.e_conv63(torch.cat([x21, x53], 1)))
        r3 = self.sigmoid(self.e_conv73(torch.cat([x11, x63], 1)))

        # x14 = self.relu(self.e_conv14(x_down))
        # x24 = self.relu(self.e_conv24(x14))
        # x34 = self.relu(self.e_conv34(x24))
        # x44 = self.relu(self.e_conv44(x31))
        # x54 = self.relu(self.e_conv54(torch.cat([x31, x44], 1)))
        # x64 = self.relu(self.e_conv64(torch.cat([x21, x54], 1)))
        # r4 = self.sigmoid(self.e_conv74(torch.cat([x11, x64], 1)))

        x_r = torch.cat([r2, r3], 1)

        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = nn.functional.interpolate(x_r, size=x.size()[2:], mode='bilinear', align_corners=True)
        r2, r3 = torch.split(x_r, 1, dim=1)
        l_ldr = torch.pow(lum + 0.00001, 1)
        l_ldr = r2 / (r2 + l_ldr)
        l_ldr = 1 - torch.pow(l_ldr + 0.00001, 2 * r3)
        ldr = torch.pow(x / rgb2y(x) + 0.00001, 0.5) * l_ldr
        return ldr, x_r



