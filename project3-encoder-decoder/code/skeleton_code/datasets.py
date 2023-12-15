import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class Loader(Dataset):

    def _check_exists(self):
        return os.path.exists(os.path.join(self.VOC2012_dir, "JPEGImages" )) and \
               os.path.exists(os.path.join(self.VOC2012_dir, "SegmentationObject"))

    def VOCdataloader(self, index):
        # Load Image
        path1 = os.path.join(self.VOC2012_dir, "JPEGImages", self.imgnames[index].split("\n")[0] + ".jpg")
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        ori_img = self.transform(img1)
        ori_img = np.asarray(ori_img)

        # Load mask(=label)
        path2 = os.path.join(self.VOC2012_dir, "SegmentationClass", self.imgnames[index].split("\n")[0] + ".png")
        img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        seg_img = cv2.resize(img2, dsize=(self.resize, self.resize), interpolation=cv2.INTER_NEAREST)

        label = np.zeros((self.resize, self.resize), dtype=np.uint8)
        for i in range(self.resize):
            for j in range(self.resize):
                label[i][j] = self.cls[tuple(seg_img[i, j, :])]

        t_ori_img = torch.from_numpy(ori_img)
        label1 = torch.Tensor(label)

        return t_ori_img, label1

    def __init__(self, VOC2012_dir, flag, resize, transforms):
        self.VOC2012_dir = VOC2012_dir
        self.resize = resize
        self.transform = transforms
        self.flag = flag

        with open(self.VOC2012_dir + "/ImageSets/Segmentation/trainval.txt", 'r') as f:
            self.lines = f.readlines()

        # Split Test and Train
        self.fold = int(len(self.lines)*0.2)

        if self.flag == 'train':
            # self.imgnames = self.lines[:50] # Tip : you can adjust the number of images and run the quickly during debugging.
            self.imgnames = self.lines[self.fold:]

        else:
            # self.imgnames = self.lines[50:60] # Tip : you can adjust the number of images and run the quickly during debugging.
            self.imgnames = self.lines[:self.fold]

        self.cls = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2,  # 0:background, 1:aeroplane, 2:bicycle
               (128, 128, 0): 3, (0, 0, 128): 4, (128, 0, 128): 5,  # 3:bird, 4:boat, 5:bottle
               (0, 128, 128): 6, (128, 128, 128): 7, (64, 0, 0): 8,  # 6:bus, 7:car, 8:cat
               (192, 0, 0): 9, (64, 128, 0): 10, (192, 128, 0): 11,  # 9:chair, 10:cow, 11:diningtable
               (64, 0, 128): 12, (192, 0, 128): 13, (64, 128, 128): 14,  # 12:dog, 13:horse, 14:motorbike
               (192, 128, 128): 15, (0, 64, 0): 16, (128, 64, 0): 17,  # 15:person, 16:pottedplant, 17:sheep
               (0, 192, 0): 18, (128, 192, 0): 19, (0, 64, 128): 20,  # 18:sofa, 19:train, 20:tvmonitor
               (224, 224, 192): 21}  # edge

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")


    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        images, masks = self.VOCdataloader(index)

        return images, masks

