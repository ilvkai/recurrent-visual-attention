
from __future__ import print_function, absolute_import

import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from utils.data.my_utils import read_image

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

from config import dreyeve_dir, tmp_dir
from config import dreyeve_train_seq, dreyeve_test_seq
from config import train_frame_range, val_frame_range, test_frame_range, frames_per_seq
from config import w, h
import cv2

class Dreyeve(data_utl.Dataset):
    def __init__(self, phase):
        # self.split_file = split_file
        self.root = tmp_dir
        self.phase = phase
        self.debug = False

        if self.phase == 'train':
            self.sequences = dreyeve_train_seq
            self.allowed_frames = train_frame_range
            self.allow_mirror = True
        elif self.phase == 'val':
            self.sequences = dreyeve_train_seq
            self.allowed_frames = val_frame_range
            self.allow_mirror = False
        elif self.phase == 'test':
            self.sequences = dreyeve_test_seq
            self.allowed_frames = test_frame_range
            self.allow_mirror = False

        # generate batch signatures
        self.data = self.make_dreye_dataset(self.sequences, self.allowed_frames)
        self.mean_imgae_256 = read_image(os.path.join(dreyeve_dir, 'dreyeve_mean_frame.png'),
                                                channels_first=True, resize_dim=(w,h))
        self.load_mode = 1

    def make_dreye_dataset(self, sequences, allowed_frames):
        """
        Function to create dataset for the Dreyeve dataset.
        """
        dataset = []
        for indexSeq in sequences:
            # print('process seq')
            for indexFrame in allowed_frames:
                dataset.append((indexSeq, indexFrame - frames_per_seq + 1, indexFrame))
        return dataset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        indexSequence, frameBegin, frameEnd = self.data[index]
        imgs_resize, fix_h_w = self.load_frames(indexSequence, frameBegin, frameEnd)

        # last_frame =
        path_last_frame = os.path.join(dreyeve_dir, '{:02d}'.format(indexSequence), 'frames',
                                       '{:06d}.jpg'.format(frameEnd))
        ptLastFrame = read_image(path_last_frame, channels_first=True, resize_dim=(w,h))

        loc = self.get_mean_loc(indexSequence, frameEnd)


        # return torch.from_numpy(imgs_resize),  torch.from_numpy(ptLastFrame), \
        #        torch.from_numpy(fix_h_w)
        return torch.from_numpy(imgs_resize),  torch.from_numpy(fix_h_w), torch.from_numpy(loc),\
               indexSequence, frameEnd

    def __len__(self):
        return len(self.data)

    def load_frames(self, indexSequence, frameBegin, frameEnd):
        # init imgs
        imgs_h_w = np.zeros(shape=(3, frames_per_seq, h, w), dtype=np.float32)

        # get fixation maps
        pathFix = os.path.join(self.root, '{:02d}'.format(indexSequence), 'saliency_fix', '{:06d}.png'.format(frameEnd))
        fixationMap = read_image(pathFix, channels_first=True, color=False, resize_dim=(h, w)) / 255

        for indexFrame in range(frameBegin, frameEnd + 1):
            pathImg = os.path.join(self.root, '{:02d}'.format(indexSequence), 'frames-256',
                                   '{:06d}.jpg'.format(indexFrame))
            singleImgResize = read_image(pathImg, channels_first=True, resize_dim=(h,w))

            imgs_h_w[:, indexFrame - frameBegin, :, :] = singleImgResize/255

        return imgs_h_w, fixationMap

    def get_mean_loc(self, indexSeq, indexFrame):
        # the location ranges from -1 to 1 and (x, y)
        pathFix = os.path.join(self.root, '{:02d}'.format(indexSeq), 'saliency_fix', '{:06d}.png'.format(indexFrame))
        fixationMap = read_image(pathFix, channels_first=True, color=False) / 255
        loc = np.mean(np.where(fixationMap ==np.max(fixationMap)), axis=1)
        loc = loc/[fixationMap.shape[0:2]] * 2 -1
        # cv2.imread('temp.jpg',fixationMap)

        # pt_loc = (int(loc[0]), int(loc[1]))
        return loc





