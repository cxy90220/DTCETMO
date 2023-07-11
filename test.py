import torch
import torchvision
import torch.optim
import os
import model
import numpy as np
import glob
import time
import imageio
import cv2


def dataloader(path):
	data_hdr = np.asarray(imageio.v2.imread(path, format='HDR-FI'))
	# data_hdr = cv2.resize(data_hdr, (1024, 1024))
	data_hdr = torch.from_numpy(data_hdr).float()
	data_hdr = data_hdr.permute(2, 0, 1)
	data_hdr = data_hdr.cuda().unsqueeze(0)
	return data_hdr


def hdr2sdr(image_path):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	scale_factor = 32
	tmo_net = model.Tmonet(scale_factor).cuda()
	tmo_net = torch.nn.DataParallel(tmo_net).cuda()
	tmo_net.load_state_dict(torch.load('snapshots/Epoch399.pth'))

	img_hdr = dataloader(image_path)
	start = time.time()
	img_ldr, _ = tmo_net(img_hdr)
	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('testdata', 'result')
	image_path = image_path.replace('hdr', 'png')
	result_path = image_path
	torchvision.utils.save_image(img_ldr, result_path)


if __name__ == '__main__':
	with torch.no_grad():
		filePath = 'data/testdata'
		test_list = glob.glob(filePath+"/*hdr")
		for image in test_list:
			print(image)
			hdr2sdr(image)
