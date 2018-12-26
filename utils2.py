import json
import pickle
import os,os.path
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle

def train_val_split():
    addr = '/home/disk3/wcf/kitti_dataset/'
    split_c = ['calib','image_2','label_2']
    with open("val_list.txt","rb") as pfile:
        val_list = pickle.load(pfile)
    for split in split_c:
        for valItem in val_list:
            itemAddr = os.path.join(addr,'data_object_'+split,'training',split,str(valItem)+'.*')
            desAddr = os.path.join(addr,'data_object_'+split,'validation',split)
            os.system('mv '+ itemAddr + ' '+desAddr)


def list2file(addr, listt):
    with open(addr, "w") as file:
        for item in listt:
            file.write("%s\n"%item)

def file2list(addr):
    with open(addr,"r") as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        return content

def generate_list_binary(attri):
    label_dir = join('/home/wcf/projects/kitt_act/dataset',attri,'calib')
    root_dir = '/home/wcf/projects/kitt_act/'
    labels = []
    for label_file in sorted(os.listdir(label_dir)):
        labels.append(label_file[:label_file.find('.')])
    with open(join(root_dir, attri+"_list.txt"), "wb") as pfile:
        pickle.dump(labels, pfile)

def check_generated_list(attri):
    label_dir = join('/home/disk3/wcf/kitti_dataset/real_dataset/new_dataset',attri,'label_2')
    root_dir = '/home/wcf/projects/kitt_act/'
    labels = set()
    for label_file in sorted(os.listdir(label_dir)):
        labels.add(label_file[:label_file.find('_')])
    list2file(join(root_dir,'new_'+attri+"_list.txt"),sorted(list(labels)))

def  get_real_list(attri):
    fileDir = attri + '_list.txt'
    outFileDir = attri + '_list_txt.txt'
    with open(fileDir,'rb') as pfile:
        attriList = pickle.load(pfile)
    list2file(outFileDir,attriList)

def get_training_list():
    root_dir = '/home/disk3/wcf/kitti_dataset/real_dataset/new_dataset/train/image_2'
    all_training_set = set()
    for label_file in sorted(os.listdir(root_dir)):
        all_training_set.add(label_file[:label_file.find('_')+2])
    all_training_list = sorted(list(all_training_set))
    list2file(join('/home/wcf/projects/kitt_act/','all_training_list.txt'),all_training_list)


def read_label_file(addr):
    '''
    read the label file and return the three value x,y,z in the form of list
    '''
    coord_list = file2list(addr)[0].strip().split(' ')
    x,y,z = float(coord_list[0]),float(coord_list[1]),float(coord_list[2])
    return np.array([x,y,z])

def cal_mean_L2_dis():
    iMax = 64
    jMax = 10
    root_dir = '/home/disk3/wcf/kitti_dataset/real_dataset/new_dataset/train/label_2'
    all_training_list = file2list('all_training_list.txt') # all the training list with %06d_%01d imageId and carId
    
    #all_diff_array = np.array()
    all_diff_array = np.array([0])
    count = 0
    total = len(all_training_list)
    for image_list in all_training_list:
        if count % 100 == 0:    
            startTime = time.time()
        for i in range (iMax):
            for j in range(jMax - 1):
                label_dir = join(root_dir, image_list + '_%03d_%01d.txt'%(i,j))
                label_dir_next = join(root_dir, image_list + '_%03d_%01d.txt'%(i,j+1))
                coord_now = read_label_file(label_dir)
                coord_next = read_label_file(label_dir_next)
                tmp_l2_diff = np.linalg.norm(coord_now - coord_next)
                all_diff_array = np.append(all_diff_array,tmp_l2_diff)
        count += 1
        finish_rate = float(count)/total * 100
        if count % 100 == 99:
            endTime = time.time()
            diffTime = float(endTime - startTime)/60    # in minute
    #         print "Finshed %.2f%% , estimate remaining time = %d minutes" %(finish_rate, diffTime/100 * (total-count))

    # print " the min is:", all_diff_array[1:].min()
    # print " the max is:", all_diff_array[1:].max()
    # print " the mean is :", all_diff_array[1:].mean()
    fig = plt.hist(all_diff_array[1:], normed=0)
    plt.title('L2 distribution')
    plt.xlabel('diff')
    plt.ylabel('Freq')
    plt.savefig('distri.png')


def allLabels2npy(attri):
    iMax = 64
    jMax = 10
    root_dir = '/home/disk3/wcf/kitti_dataset/real_dataset/new_dataset/train/label_2'
    all_training_list = file2list(attri + '_training_list.txt') # all the training list with %06d_%01d imageId and carId
    count = 0
    total = len(all_training_list)
    allLabels = np.array([[0,0,0]])
    for image_list in all_training_list:
        if count % 100 == 0:
            startTime = time.time()
        for i in range(iMax):
            for j in range(jMax):
                label_dir = join(root_dir, image_list + '_%03d_%01d.txt'%(i,j))
                coord = read_label_file(label_dir)
                allLabels = np.vstack((allLabels, coord))
        count += 1
        finish_rate = float(count)/total * 100
        if count % 100 == 99:
            endTime = time.time()
            diffTime = float(endTime - startTime)/60    # in minute
            #print "Finshed %.2f%% , estimate remaining time = %d minutes" %(finish_rate, diffTime/100 * (total-count))

    np.save(attri + '_allLabels.npy', allLabels[1:])


def get_small_dataset():
    all_training_list = file2list('all_training_list.txt')
    shuffle(all_training_list)
    all_training_list = sorted(all_training_list[:100])
    list2file('small_training_list.txt', all_training_list)

def get_npy_index(image_id,i,j):
    assert i<64, "Input sample id >=64"
    assert j<10, "Input interp id >=10"
    return image_id * 640 + i * 10 + j

def get_one_dataset():
    one_training_list = file2list('all_training_list.txt')
    shuffle(one_training_list)
    one_training_list = sorted(one_training_list[0:1])
    list2file('one_training_list.txt', one_training_list)

if __name__ == '__main__':
    # get_real_list('train')
    # generate_list_binary('train')
    #fileList2file('train')
    # fileList2file('validation')
    # check_generated_list('train')
    #cal_mean_L2_dis()
    #print (read_label_file('000038_0_000_1.txt'))
    allLabels2npy('one')
    # get_small_dataset()
    # get_one_dataset()
