import numpy as np
from scipy.signal.windows import gaussian
import torch
import math
import torch.nn.functional as fun


def rgb2y(img):
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.2126*r + 0.7152*g + 0.0722*b
    return y


def structuralfidelity(l_hdr, l_ldr, level, weight, window):
    f = 2 ** level
    s_local = []
    s_maps = []
    for _ in range(level):
        f = f / 2
        sl, sm = slocal(l_hdr, l_ldr, window, f)
        s_local.append(sl)
        s_maps.append(sm)
        l_hdr = fun.avg_pool2d(l_hdr, (2, 2))
        l_ldr = fun.avg_pool2d(l_ldr, (2, 2))
    s_local = torch.stack(s_local)
    s = torch.prod(torch.pow(s_local, weight), 0)
    return s, s_local, s_maps


def slocal(img1, img2, window, sf, c1=0.01, c2=10.):
    mu1 = fun.conv2d(img1, window)
    mu2 = fun.conv2d(img2, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = fun.conv2d(img1 * img1, window) - mu1_sq
    sigma2_sq = fun.conv2d(img2 * img2, window) - mu2_sq
    relu = torch.nn.ReLU(inplace=True)
    sigma1 = torch.sqrt(relu(sigma1_sq))
    sigma2 = torch.sqrt(relu(sigma2_sq))
    sigma12 = fun.conv2d(img1 * img2, window) - mu1_mu2

    csf = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(- (0.114 * sf) ** 1.1)
    u = 128 / (1.4 * csf)
    sig = u / 3.
    sigma1p = 0.5 * (1 + torch.erf((sigma1 - u) / (sig * math.sqrt(2.0))))
    sigma2p = 0.5 * (1 + torch.erf((sigma2 - u) / (sig * math.sqrt(2.0))))
    s_map = ((2 * sigma1p * sigma2p + c1) / (sigma1p ** 2 + sigma2p ** 2 + c1)
             * ((sigma12 + c2) / (sigma1 * sigma2 + c2)))
    s = torch.mean(s_map, [2, 3], keepdim=True)
    return s, s_map


def sigmawin(img):
    factor = 11
    avg1 = torch.nn.functional.interpolate(img, scale_factor=1./factor, mode='area')
    avg2 = torch.nn.functional.interpolate(avg1, size=img.size()[2:], mode='area')
    sig = torch.sqrt(torch.nn.functional.interpolate(torch.pow(img - avg2, 2), scale_factor=1./factor, mode='area') + 0.00001)
    sigavg = torch.mean(sig, [2, 3], keepdim=True)
    return sigavg


def beta1(x):
    result = 5009.72154 * torch.pow(x, 3.4) * torch.pow(1 - x, 9.1)
    return result


def norm1(x, u, sig):
    x = (x - u) / sig
    result = 0.39894 / sig * torch.exp(-torch.pow(x, 2) / 2)
    return result


def statisticalnaturalness(l_ldr):
    l_ldr = l_ldr
    muhat = torch.tensor(115.94)
    sigmahat = 27.99
    u = torch.mean(l_ldr, [2, 3], keepdim=True)
    sig = sigmawin(l_ldr)
    c = beta1(sig / 64.29)
    pc = c / 3.3323
    # pc = c / beta1(torch.tensor(0.3))
    b = norm1(u, muhat, sigmahat)
    pb = b / 0.014253
    # pb = b / norm1(torch.tensor(153.), torch.tensor(153.), sigmahat)
    n = pb * pc
    return n


class TMQI(torch.nn.Module):
    def __init__(self):
        super(TMQI, self).__init__()

    def forward(self, hdrimage, ldrimage):
        lvl = 5
        weight = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda()
        weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gauss = gaussian(11, 1.5)
        window = torch.tensor(np.outer(gauss, gauss)).unsqueeze(0).unsqueeze(0)
        window = (window/window.sum()).cuda()
        l_hdr = rgb2y(hdrimage)
        l_ldr = rgb2y(ldrimage)
        n = statisticalnaturalness(l_ldr)
        l_ldr = l_ldr.double()
        l_hdr = l_hdr.double()
        factor = 2 ** 32 - 1.
        lmin = torch.min(torch.reshape(l_hdr, (l_hdr.size()[0], 1, -1)), 2)[0].unsqueeze(-1).unsqueeze(-1)
        lmax = torch.max(torch.reshape(l_hdr, (l_hdr.size()[0], 1, -1)), 2)[0].unsqueeze(-1).unsqueeze(-1)
        l_hdr = factor * (l_hdr - lmin) / (lmax - lmin)
        s, s_local, s_maps = structuralfidelity(l_hdr, l_ldr, lvl, weight, window)
        a = 0.8012
        b = 0.3046
        c = 0.7088
        q = a * (s ** b) + (1. - a) * (n ** c)
        loss = torch.mean(1.0 - q)
        return loss
