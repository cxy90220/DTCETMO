# Only considering cases where the number of pixels is less than 2^18
import torch
from torch.fft import fft2, ifft2
import math


class FSITM(torch.nn.Module):
    def __init__(self, theta, loggabor):
        super(FSITM, self).__init__()
        self.relu = torch.nn.ReLU()
        self.theta = theta
        self.loggabor = loggabor

    def forward(self, ldrimg, hdrimg):
        logh = torch.log(hdrimg+0.00001)
        lmax = torch.max(torch.reshape(logh, (logh.size()[0], 1, -1)), 2)[0].unsqueeze(-1).unsqueeze(-1)
        lmin = torch.min(torch.reshape(logh, (logh.size()[0], 1, -1)), 2)[0].unsqueeze(-1).unsqueeze(-1)
        logh = ((logh - lmin) * 255. / (lmax - lmin))
        phaseh = phasecong100(logh, self.theta, self.loggabor)
        phasel = phasecong100(ldrimg, self.theta, self.loggabor)
        q = torch.mean(10000*(-self.relu((-self.relu(phasel*phaseh) + 0.0001)) + 0.0001))
        return 1 - q


def phasecong100(img, theta, loggabor):

    imagefft = fft2(img).cuda()
    eo = dict()
    energyv = torch.zeros((img.size()[0], img.size()[1], img.size()[2], img.size()[3], 3)).cuda()
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)
    for o in range(0, 2):
        angl = o * math.pi / 2
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread = (torch.cos(dtheta) + 1.) / 2.
        sume = torch.zeros((img.size()[2], img.size()[3])).unsqueeze(0).unsqueeze(0).cuda()
        sumo = torch.zeros((img.size()[2], img.size()[3])).unsqueeze(0).unsqueeze(0).cuda()
        for s in range(0, 2):
            filter_ = loggabor[s] * spread
            eo[(s, o+1)] = ifft2(imagefft * filter_)
            sume = sume + torch.real(eo[(s, o + 1)])
            sumo = sumo + torch.imag(eo[(s, o + 1)])
        energyv[:, :, :, :, 0] = energyv[:, :, :, :, 0] + sume
        energyv[:, :, :, :, 1] = energyv[:, :, :, :, 1] + math.cos(angl) * sumo
    oddv = torch.sqrt(energyv[:, :, :, :, 0] ** 2 + energyv[:, :, :, :, 1] ** 2)
    feattype = torch.atan2(energyv[:, :, :, :, 0], oddv)
    return feattype
