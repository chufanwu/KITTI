# dataloader.py
import os
import sys
import numpy as np
from PIL import Image
import pickle
from utils2 import *


datasetDir = '/home/disk3/wcf/kitti_dataset/real_dataset/new_dataset/'
attri = 'all'


# the dataloader of new images
class KittiDataset(object):
    def __init__(self, img_dir=join(datasetDir,'train/image_2'), label_dir='./' + attri + '_allLabels.npy'):
        self.img_dir = img_dir
        self.label_npy = np.load(label_dir)
        self.dataset = file2list(attri + '_training_list.txt')
        self.n_seq = len(self.dataset)
        self.n_sam = 64
        self.seqlen = 10
        self.lookahead = 3



    def __iter__(self):
        return self


    def __next__(self):
        # randomly choose a specific car and specific sample(from 1/64)
        frame_list = []
        ground_list = []
        seq_id = np.random.randint(self.n_seq)
        sam_id = np.random.randint(self.n_sam)
        start_idx =  0 # the ground truth of the seqs
        end_idx = min((start_idx + 2 * self.lookahead + np.random.randint(self.lookahead + 2)), self.seqlen)
        for idx in reversed(range(start_idx, end_idx)):
            img_path = join(self.img_dir, self.dataset[seq_id]+"_%03d_%01d.png" %(sam_id,idx))
            coord = self.label_npy[get_npy_index(seq_id,sam_id,idx)]
            frame_list.append(img_path)
            ground_list.append(coord)
        return frame_list, ground_list, end_idx - start_idx

    

    next = __next__



# the dataloader for initialize actions
class InitActionDataset(object):
    '''
    input the car id and output 64 images with their following distances
    '''

    def __init__(self, init_actor_batch=64, img_dir=join(datasetDir,'train/image_2'), label_dir='./' + attri + '_allLabels.npy'):
        self.init_actor_batch = init_actor_batch
        self.img_dir = img_dir
        self.label_npy = np.load(label_dir)
        self.dataset = file2list(attri + '_training_list.txt')
        self.n_seq = len(self.dataset)
        self.n_sam = 64
        self.seqlen = 10




    def __iter__(self):
        return self


    def __next__(self, car_id):
        # return 64 frame list and the following distance list
        frame_list = []
        dis_numpy = np.zeros([self.init_actor_batch, 3])

        seq_id = self.dataset.index(car_id)
        tmp_arr = np.arange(self.n_sam * self.seqlen)
        np.random.shuffle(tmp_arr)  # shuffle 0~639
        count = 0
        for i in tmp_arr:
            if i % self.seqlen == 0:
                continue
            else:
                count += 1
                sam_id = int(i / self.seqlen)
                idx = int(i % self.seqlen)
                coord = self.label_npy[get_npy_index(seq_id,sam_id,idx)]
                coord_prev = self.label_npy[get_npy_index(seq_id,sam_id,idx-1)]
                dis = coord_prev - coord
                img_path = join(self.img_dir, self.dataset[seq_id]+"_%03d_%01d.png" %(sam_id,idx))
                frame_list.append(img_path)
                dis_numpy[count-1] = dis
                if count == self.init_actor_batch:
                    break

        return frame_list, dis_numpy
    

    next = __next__