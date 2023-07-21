import torch
import torch.optim
import os
import argparse
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR, SequentialLR
import FSITM
import dataloader
import model
import TMQI
from torch.fft import ifftshift


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def lowpass_filter(size=None, cutoff=None, n=None):

    if type(size) == int:
        rows = cols = size
    else:
        rows, cols = size
    x_range = torch.linspace(-0.5, 0.5, cols)
    y_range = torch.linspace(-0.5, 0.5, rows)
    x, y = torch.meshgrid(x_range, y_range)
    radius = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).cuda()
    f = ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))
    return f


def fsitmpara(imgsize):
    x_range = torch.linspace(-0.5, 0.5, imgsize)
    y_range = torch.linspace(-0.5, 0.5, imgsize)
    x, y = torch.meshgrid(x_range, y_range)
    radius = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).cuda()
    theta = torch.atan2(- y, x).unsqueeze(0).unsqueeze(0).cuda()
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1.
    lp = lowpass_filter((imgsize, imgsize), 0.45, 15)
    loggabor = []
    for s in range(0, 2):
        fo = 1.0 / (4 ** s)
        loggabor.append(torch.exp((- (torch.log(radius / fo)) ** 2) / 0.37115))
        loggabor[-1] *= lp
        loggabor[-1][0, 0, 0, 0] = 0
    return theta, loggabor


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 32
    w = config.loss_weight
    tmo_net = model.Tmonet(scale_factor).cuda()
    tmo_net.apply(weights_init)
    tmo_net = torch.nn.DataParallel(tmo_net).cuda()
    if config.load_pretrain:
        tmo_net.load_state_dict(torch.load(config.pretrain_dir))
    imgsize = config.train_imgsize
    train_dataset = dataloader.Hdrloader(config.images_path, imgsize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    ltmqi = TMQI.TMQI()
    theta, loggabor = fsitmpara(imgsize)
    lfsitm = FSITM.FSITM(theta, loggabor)

    optimizer = torch.optim.Adam(tmo_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    tmo_net.train()
    loss_list = []
    start = time.time()

    schedulers = [CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2),
                  ConstantLR(optimizer, factor=0.01, total_iters=400, last_epoch=-1)]
    scheduler = SequentialLR(optimizer, schedulers, milestones=[300])

    for epoch in range(config.num_epochs):
        print("epoch", epoch)
        loss_sum = 0
        loss_sumt = 0
        loss_sumf = 0
        for iteration, img_hdr in enumerate(train_loader):
            img_hdr = img_hdr.cuda()
            img_ldr, p = tmo_net(img_hdr)
            loss_tmqi = (ltmqi(img_hdr, (img_ldr*255)))
            rl, gl, bl = torch.split(img_ldr * 255, 1, dim=1)
            rh, gh, bh = torch.split(img_hdr, 1, dim=1)
            loss_fsitm = (lfsitm(rl, rh) + lfsitm(gl, gh) + lfsitm(bl, bh)) / 3
            loss = w*loss_tmqi + (2.0-w)*loss_fsitm
            # if torch.isnan(loss_tmqi).any():
            #     optimizer.zero_grad()
            #     continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tmo_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sumt = loss_sumt + loss_tmqi.item()
            loss_sumf = loss_sumf + loss_fsitm.item()
            loss_sum = loss_sum + loss.item()
        loss_list.append(loss_sum)
        print("Loss at epoch", epoch, ":", loss_sum, loss_sumt, loss_sumf)
        torch.save(tmo_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
    end_time = (time.time() - start)
    print(end_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default="data/traindata/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch399.pth")
    parser.add_argument('--train_imgsize', type=int, default=512)
    parser.add_argument('--loss_weight', type=float, default=1.5)

    configure = parser.parse_args()
    if not os.path.exists(configure.snapshots_folder):
        os.mkdir(configure.snapshots_folder)
    train(configure)
