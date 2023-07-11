import imageio
import torch
import torch.utils.data as data
import numpy as np
import glob
import random
import cv2
random.seed(1258)


def populate_train_list(lowlight_images_path):
	image_list = glob.glob(lowlight_images_path + "*.hdr")
	random.shuffle(image_list)
	return image_list


class Hdrloader(data.Dataset):

	def __init__(self, images_path, imgsize):
		self.train_list = populate_train_list(images_path)
		self.size = imgsize
		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

	def __getitem__(self, index):
		data_path = self.data_list[index]
		data_hdr = imageio.v2.imread(data_path, format='HDR-FI')
		data_hdr = np.asarray(data_hdr)
		data_hdr = cv2.resize(data_hdr, (self.size, self.size))
		data_hdr = torch.from_numpy(data_hdr).float()
		data_hdr = data_hdr.permute(2, 0, 1)
		return data_hdr

	def __len__(self):
		return len(self.data_list)

