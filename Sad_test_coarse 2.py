import cv2, os
import time
import numpy as np
import copy
from scipy.misc import imresize

from model import *

import torch
import torch.optim as optim
from torch.autograd import Variable

BIN, OVERLAP = 4, 0.1
VEHICLES = ['Car'] # , 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']
# % index for painting
point_idx = np.mat([[1, 3, 6, 8]])
edge_idx = np.mat([[2, 4, 5], [2, 4, 7], [2, 5, 7], [4, 5, 7]])
show_flag = 0

def compute_anchors(angle):
    anchors = []

    wedge = 2. * np.pi / BIN
    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])

    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])

    return anchors


def parse_anno(label_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in sorted(os.listdir(label_dir)):
        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))
            image_file = label_file.replace('txt', 'png')
            if line[0] in VEHICLES and truncated < 0.01 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi / 2.
                if new_alpha < 0:
                    # print "new_alpha is really < 0 :", new_alpha, " and picture:", label_file
                    new_alpha = new_alpha + 2. * np.pi
                    # print "now new_alpha is add 2 pi:", new_alpha, " and will be subtracted by:", int(
                    #     new_alpha / (2. * np.pi)) * (2. * np.pi)
                if int(new_alpha / (2. * np.pi)) * (2. * np.pi) != 0:
                    print "will be subtracted by:", int(new_alpha / (2. * np.pi)) * (2. * np.pi)
                new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)  # rubbish! it is zero forever!

                obj = {'name': line[0],
                       'image': image_file,
                       'alpha': float(line[3]),  #  [-pi, pi]
                       'r_y': float(line[14]),  #  [-pi, pi]
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'real_dims': np.array([float(number) for number in line[8:11]]),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'xyz': np.array([float(number) for number in line[11:14]]),
                       'new_alpha': new_alpha
                       }

                dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                if obj['ymax'] - obj['ymin'] < 40:
                    continue

                all_objs.append(obj)
    ###### flip data
    for obj in all_objs:
        # Fix dimensions
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(obj['new_alpha'])

        # print anchors

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Fix orientation and confidence for flip
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(2. * np.pi - obj['new_alpha'])
        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence

    return all_objs


def gen_samples(obj, image_dir, calib_dir):
    n_pos = 32
    n_neg = 96
    pos_regions = np.zeros((n_pos, 107, 107, 3), dtype = 'uint8')
    neg_regions = np.zeros((n_neg, 107, 107, 3), dtype = 'uint8')
    image_file = obj['image']
    img = cv2.imread(image_dir + image_file)

    label_file = image_file.replace('png', 'txt')
    for line in open(calib_dir + label_file).readlines():
        line = line.strip().split(' ')
        if line[0] == 'P2:':
            Rt = np.mat([[float(line[1]),float(line[2]),float(line[3]),float(line[4])], \
                        [float(line[5]),float(line[6]),float(line[7]),float(line[8])], \
                        [float(line[9]),float(line[10]),float(line[11]),float(line[12])],[0,0,0,1]])  # 4*4

    # compute rotational matrix around yaw axis
    R = np.mat(
        [[np.cos(obj['r_y']), 0, np.sin(obj['r_y'])], [0, 1, 0], [-np.sin(obj['r_y']), 0, np.cos(obj['r_y'])]])

    # 3D bounding box dimensions
    h = obj['real_dims'][0]
    w = obj['real_dims'][1]
    l = obj['real_dims'][2]

    # 3D bounding box corners
    x_corners = np.mat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.mat([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.mat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    Corners = np.vstack((x_corners, y_corners, z_corners))

    # samples corners
    trans_f = 0.05
    n = 32*3
    sample_type = 'gaussian'
    flag = 0
    while(flag==0):
        if sample_type == 'gaussian':
            trans = trans_f * np.mean([w, l]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
        samples = np.zeros((n,3,8))
        x = np.zeros(n)
        z = np.zeros(n)
        iou = np.zeros(n)
        for i in range(n):
            samples[i, :, :] = np.vstack((x_corners + trans[i, 0], y_corners, z_corners + trans[i, 1]))
            x[i] = np.abs(trans[i, 0])
            z[i] = np.abs(trans[i, 1])
            iou[i] = (w * l - w * z[i] - x[i] * l + z[i] * x[i]) / (w * l + w * z[i] + x[i] * l - z[i] * x[i])
        pos_index = np.where(iou >= 0.85)
        if pos_index[0].shape[0] >= n_pos:
            flag = 1

    pos_samples_tmp = samples[pos_index,:,:]
    pos_samples = pos_samples_tmp[0,0:n_pos,:,:]
    for i in range(n_pos):
        sam_2D = project3d2d(pos_samples[i, :, :], R, obj, Rt)
        img_tmp = copy.deepcopy(img)
        img_tmp = draw3d(sam_2D, img_tmp, (0, 255, 0))
        if i == 0 and show_flag == 1:
            cv2.imshow('image', img_tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # compare the 3d bounding rect and 2d rect
        d3xmin = int(sam_2D[0, :].min())
        d3ymin = int(sam_2D[1, :].min())
        d3xmax = int(sam_2D[0, :].max())
        d3ymax = int(sam_2D[1, :].max())

        # add contour
        d3ymin = d3ymin - int((d3ymax - d3ymin) * 0.05)
        d3ymax = d3ymax + int((d3ymax - d3ymin) * 0.05)
        d3xmin = d3xmin - int((d3xmax - d3xmin) * 0.05)
        d3xmax = d3xmax + int((d3xmax - d3xmin) * 0.05)
        if d3ymin < 0: d3ymin = 0
        if d3xmin < 0: d3xmin = 0
        if d3ymax > img.shape[0]: d3ymax = img.shape[0]
        if d3xmax > img.shape[1]: d3xmax = img.shape[1]
        d3patch = img_tmp[d3ymin:d3ymax, d3xmin:d3xmax]
        d3patch = imresize(d3patch, (107, 107))
        pos_regions[i] = d3patch

    # for negative samples
    trans_f = 0.4
    n = 256*3
    sample_type = 'uniform'
    flag = 0
    while(flag==0):
        if sample_type == 'uniform':
            trans = trans_f * np.mean([w, l]) * (np.random.rand(n,2)*2-1)
        samples = np.zeros((n,3,8))
        x = np.zeros(n)
        z = np.zeros(n)
        iou = np.zeros(n)
        for i in range(n):
            samples[i, :, :] = np.vstack((x_corners + trans[i, 0], y_corners, z_corners + trans[i, 1]))
            x[i] = np.abs(trans[i, 0])
            z[i] = np.abs(trans[i, 1])
            iou[i] = (w * l - w * z[i] - x[i] * l + z[i] * x[i]) / (w * l + w * z[i] + x[i] * l - z[i] * x[i])
        neg_index = np.where(iou <= 0.6)
        if neg_index[0].shape[0] >= n_neg:
            flag = 1

    neg_samples_tmp = samples[neg_index, :, :]
    neg_samples = neg_samples_tmp[0, 0:n_neg, :, :]
    for i in range(n_neg):
        sam_2D = project3d2d(neg_samples[i, :, :], R, obj, Rt)
        img_tmp = copy.deepcopy(img)
        img_tmp = draw3d(sam_2D, img_tmp, (0, 255, 0))
        if i == 0 and show_flag == 1:
            cv2.imshow('image', img_tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # compare the 3d bounding rect and 2d rect
        d3xmin = int(sam_2D[0, :].min())
        d3ymin = int(sam_2D[1, :].min())
        d3xmax = int(sam_2D[0, :].max())
        d3ymax = int(sam_2D[1, :].max())

        # add contour
        d3ymin = d3ymin - int((d3ymax - d3ymin) * 0.05)
        d3ymax = d3ymax + int((d3ymax - d3ymin) * 0.05)
        d3xmin = d3xmin - int((d3xmax - d3xmin) * 0.05)
        d3xmax = d3xmax + int((d3xmax - d3xmin) * 0.05)
        if d3ymin < 0: d3ymin = 0
        if d3xmin < 0: d3xmin = 0
        if d3ymax > img.shape[0]: d3ymax = img.shape[0]
        if d3xmax > img.shape[1]: d3xmax = img.shape[1]
        d3patch = img_tmp[d3ymin:d3ymax, d3xmin:d3xmax]
        d3patch = imresize(d3patch, (107, 107))
        neg_regions[i] = d3patch

    pos_regions = pos_regions.transpose(0,3,1,2)
    pos_regions = pos_regions.astype('float32') - 128.
    neg_regions = neg_regions.transpose(0,3,1,2)
    neg_regions = neg_regions.astype('float32') - 128.
    pos_regions = torch.from_numpy(pos_regions).float()
    neg_regions = torch.from_numpy(neg_regions).float()
    return pos_regions, neg_regions


def project3d2d(Corners, R, obj, Rt):
    # rotate and translate 3D bounding box
    corners_3D = R * Corners  # 3*3 3*8 -> 3*8
    corners_3D[0, :] = corners_3D[0, :] + obj['xyz'][0]
    corners_3D[1, :] = corners_3D[1, :] + obj['xyz'][1]
    corners_3D[2, :] = corners_3D[2, :] + obj['xyz'][2]

    # % only draw 3D bounding box for objects in front of the camera
    # if any(corners_3D(3,:)<0.1)
    #   corners_2D = [];
    #   return;
    # end

    # % project the 3D bounding box into the image plane
    pts_2D = Rt * np.vstack((corners_3D, [1, 1, 1, 1, 1, 1, 1, 1]))  # 4*4 4*8 -> 4*8
    # % scale projected points
    pts_2D[0, :] = np.divide(pts_2D[0, :], pts_2D[2, :])
    pts_2D[1, :] = np.divide(pts_2D[1, :], pts_2D[2, :])

    return pts_2D


def draw3d(pts_2D, img, color):
    for f in range(4):
        for g in range(3):
            cv2.line(img, (int(pts_2D[0, point_idx[0, f] - 1]) + 1, int(pts_2D[1, point_idx[0, f] - 1]) + 1), \
                 (int(pts_2D[0, edge_idx[f, g] - 1]) + 1, int(pts_2D[1, edge_idx[f, g] - 1]) + 1), color,
                 1)
    return img


def compute_iou(A, B):
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if W <= 0 or H <= 0:
        return 0
    SA = (A[2] - A[0]) * (A[3] - A[1])
    SB = (B[2] - B[0]) * (B[3] - B[1])
    cross = W * H
    IoU = float(cross)/float(SA + SB - cross)
    return IoU


def compute_iou3d(A, B):  #  [l,h,w], [x[i],y[i],z[i]])
    if A[0] <= B[0] or A[1] <= B[1] or A[2] <= B[2]:
        return 0
    SXW = B[0] * A[2] * A[1]   # x * w * h
    SLZ = (A[0] - B[0]) *  B[2] * A[1]  # (l-x) * z * h
    SHY = (A[0] - B[0]) * (A[2] - B[2]) * B[1]  # (l-x) *(w-z) * y
    cross = A[0] * A[1] * A[2] - SXW - SLZ -SHY
    IoU = float(cross)/float(A[0] * A[1] * A[2] + SXW + SLZ + SHY)
    return IoU


image_dir = './validation/image_2/'
label_dir = './validation/label_2/'
calib_dir = './validation/calib/'
box3d_loc = './validation/result_sad1_newy/'

# create a folder for saving result
if os.path.isdir(box3d_loc) == False:
    os.mkdir(box3d_loc)

# Load image & run testing
all_image = sorted(os.listdir(image_dir))

for f in all_image:
    image_file = image_dir + f
    box2d_file = label_dir + f.replace('png', 'txt')
    box3d_file = box3d_loc + f.replace('png', 'txt')
    label_file = f.replace('png', 'txt')
    for line in open(calib_dir + label_file).readlines():
        line = line.strip().split(' ')
        if line[0] == 'P2:':
            Rt = np.mat([[float(line[1]),float(line[2]),float(line[3]),float(line[4])], \
                        [float(line[5]),float(line[6]),float(line[7]),float(line[8])], \
                        [float(line[9]),float(line[10]),float(line[11]),float(line[12])],[0,0,0,1]])  # 4*4
    # print image_file
    with open(box3d_file, 'w') as box3d:
        img = cv2.imread(image_file)
        # img = img.astype(np.float32, copy=False)

        for line in open(box2d_file):
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            obj = {'name': line[0],
                   'image': image_file,
                   'alpha': float(line[3]),  # [-pi, pi]
                   'r_y': float(line[14]),  # [-pi, pi]
                   'xmin': int(float(line[4])),
                   'ymin': int(float(line[5])),
                   'xmax': int(float(line[6])),
                   'ymax': int(float(line[7])),
                   'real_dims': np.array([float(number) for number in line[8:11]]),
                   'dims': np.array([float(number) for number in line[8:11]]),
                   'xyz': np.array([float(number) for number in line[11:14]]),
                   }

            # Sample 10*10 3D bounding boxes
            if obj['name'] == 'Car':
                # 3D bounding box dimensions
                h = obj['real_dims'][0]
                w = obj['real_dims'][1]
                l = obj['real_dims'][2]
                # 3D bounding box corners
                x_corners = np.mat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
                y_corners = np.mat([0, 0, 0, 0, -h, -h, -h, -h])
                z_corners = np.mat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
                Corners = np.vstack((x_corners, y_corners, z_corners))
                # compute rotational matrix around yaw axis
                R = np.mat(
                    [[np.cos(obj['r_y']), 0, np.sin(obj['r_y'])], [0, 1, 0],
                     [-np.sin(obj['r_y']), 0, np.cos(obj['r_y'])]])
                Num_x = int(100 / l)
                Num_y = int(100 / w)
                topIoU  = 0
                topIoU_i= 0
                topIoU_j= 0
                for i in range(Num_x):
                    for j in range(Num_y):
                        obj_tmp = copy.deepcopy(obj)
                        obj_tmp['xyz'][0] = -50 + i * l  # [-50,50] left to right
                        obj_tmp['xyz'][1] = 1.65  # 2.475 - h/2  # ground prior should be 1.65!
                        obj_tmp['xyz'][2] = 1 + j * w  # [1,201]  front
                        sam_2D = project3d2d(Corners, R, obj_tmp, Rt)
                        # compare the 3d bounding rect and 2d rect
                        d3xmin = int(sam_2D[0, :].min())
                        d3ymin = int(sam_2D[1, :].min())
                        d3xmax = int(sam_2D[0, :].max())
                        d3ymax = int(sam_2D[1, :].max())
                        Box3Dto2D = [d3xmin, d3ymin, d3xmax, d3ymax]
                        Box2D = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                        IoU   = compute_iou(Box2D,Box3Dto2D)
                        if IoU > topIoU:
                            topIoU_i = i
                            topIoU_j = j
                            topIoU = IoU
                if topIoU == 0:
                    print topIoU
                    # sam_2D = project3d2d(samples[topIoU_i,topIoU_j,:,:], R, obj, Rt)
                    # img_tmp = copy.deepcopy(img)
                    # img_tmp = draw3d(sam_2D, img_tmp, (0, 255, 0))
                    # cv2.imshow('image', img_tmp)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                obj_tmp = copy.deepcopy(obj)
                obj_tmp['xyz'][0] = -50 + topIoU_i * l  # [-50,50] left to right
                obj_tmp['xyz'][1] = 2.475 - h / 2  # ground prior
                obj_tmp['xyz'][2] = 1 + topIoU_j * w  # [1,101]  front
                x = np.abs(obj_tmp['xyz'][0] - obj['xyz'][0])
                y = np.abs(obj_tmp['xyz'][1] - obj['xyz'][1])
                z = np.abs(obj_tmp['xyz'][2] - obj['xyz'][2])
                print 'xyz:', x, y, z, 'lhw:', l, h, w, 'x/l y/h z/w:', x/l , y/h, z/w,
                iou3d = compute_iou3d([l, h, w], [x, y, z])
                print 'topIoU', topIoU, 'iou3d', iou3d

                # n_can = 1024
                # # samples corners
                # trans_fx = 1
                # trans_fy = 0.25
                # trans_fz = 2
                # # sample_type = 'gaussian'
                # # if sample_type == 'gaussian':
                # #     trans = trans_f * np.mean([w, l]) * np.clip(0.5 * np.random.randn(n_can, 2), -1, 1)
                # sample_type = 'uniform'
                # if sample_type == 'uniform':
                #     trans_x = np.insert(trans_fx * l * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_y = np.insert(trans_fy * h * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_z = np.insert(trans_fz * w * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                # x = np.zeros(n_can)
                # y = np.zeros(n_can)
                # z = np.zeros(n_can)
                # iou = np.zeros(n_can)
                # topIoU2 = 0
                # topIoU_i2 = 0
                # for i in range(n_can):
                #     IoU = 0
                #     while IoU <= topIoU/2 :
                #         obj_tmp2 = copy.deepcopy(obj_tmp)
                #         obj_tmp2['xyz'][0] = obj_tmp['xyz'][0] + trans_x[i]
                #         obj_tmp2['xyz'][1] = obj_tmp['xyz'][1] + trans_y[i]
                #         obj_tmp2['xyz'][2] = obj_tmp['xyz'][2] + trans_z[i]
                #         x[i] = np.abs(obj_tmp2['xyz'][0] - obj['xyz'][0])
                #         y[i] = np.abs(obj_tmp2['xyz'][1] - obj['xyz'][1])
                #         z[i] = np.abs(obj_tmp2['xyz'][2] - obj['xyz'][2])
                #         iou[i] = compute_iou3d([l,h,w], [x[i],y[i],z[i]])
                #         # compare the 3d bounding rect and 2d rect
                #         sam_2D = project3d2d(Corners, R, obj_tmp2, Rt)
                #         d3xmin = int(sam_2D[0, :].min())
                #         d3ymin = int(sam_2D[1, :].min())
                #         d3xmax = int(sam_2D[0, :].max())
                #         d3ymax = int(sam_2D[1, :].max())
                #         # # add contour
                #         # d3ymin = d3ymin - int((d3ymax - d3ymin) * contour /2)
                #         # d3ymax = d3ymax + int((d3ymax - d3ymin) * contour /2)
                #         # d3xmin = d3xmin - int((d3xmax - d3xmin) * contour /2)
                #         # d3xmax = d3xmax + int((d3xmax - d3xmin) * contour /2)
                #         if d3ymin < 0: d3ymin = 0
                #         if d3xmin < 0: d3xmin = 0
                #         if d3xmax <= 0: d3xmax = 1
                #         if d3ymax <= 0: d3ymax = 1
                #         if d3ymin >= img.shape[0]-1: d3ymin = img.shape[0] - 2
                #         if d3xmin >= img.shape[1]-1: d3xmin = img.shape[1] - 2
                #         if d3ymax > img.shape[0]-1: d3ymax = img.shape[0] - 1
                #         if d3xmax > img.shape[1]-1: d3xmax = img.shape[1] - 1
                #         Box3Dto2D = [d3xmin, d3ymin, d3xmax, d3ymax]
                #         IoU = compute_iou(Box2D, Box3Dto2D)
                #         if IoU <= topIoU/2 :
                #             trans_x[i] = trans_fx * l * (np.random.rand(1, 1) * 2 - 1)
                #             trans_y[i] = trans_fy * h * (np.random.rand(1, 1) * 2 - 1)
                #             trans_z[i] = trans_fx * w * (np.random.rand(1, 1) * 2 - 1)
                #         elif IoU > topIoU2 :
                #             topIoU2 = IoU
                #             topIoU_i2 = i
                #
                # obj_tmp2 = copy.deepcopy(obj)
                # obj_tmp2['xyz'][0] = obj_tmp['xyz'][0] + trans_x[topIoU_i2]
                # obj_tmp2['xyz'][1] = obj_tmp['xyz'][1] + trans_y[topIoU_i2]
                # obj_tmp2['xyz'][2] = obj_tmp['xyz'][2] + trans_z[topIoU_i2]
                # x = np.abs(obj_tmp2['xyz'][0] - obj['xyz'][0])
                # y = np.abs(obj_tmp2['xyz'][1] - obj['xyz'][1])
                # z = np.abs(obj_tmp2['xyz'][2] - obj['xyz'][2])
                # print 'xyz:', x, y, z, 'lhw:', l, h, w, 'x/l y/h z/w:', x/l , y/h, z/w,
                # iou3d2 = compute_iou3d([l, h, w], [x, y, z])
                # print 'topIoU2', topIoU2, 'iou3d2', iou3d2
                #
                #
                # n_can = 1024
                # # samples corners
                # trans_fx = 1
                # trans_fy = 0.25
                # trans_fz = 2
                # # sample_type = 'gaussian'
                # # if sample_type == 'gaussian':
                # #     trans = trans_f * np.mean([w, l]) * np.clip(0.5 * np.random.randn(n_can, 2), -1, 1)
                # sample_type = 'uniform'
                # if sample_type == 'uniform':
                #     trans_x = np.insert(trans_fx * l * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_y = np.insert(trans_fy * h * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_z = np.insert(trans_fz * w * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                # x = np.zeros(n_can)
                # y = np.zeros(n_can)
                # z = np.zeros(n_can)
                # iou = np.zeros(n_can)
                # topIoU3 = 0
                # topIoU_i3 = 0
                # for i in range(n_can):
                #     IoU = 0
                #     while IoU <= topIoU2/2 :
                #         obj_tmp3 = copy.deepcopy(obj_tmp)
                #         obj_tmp3['xyz'][0] = obj_tmp2['xyz'][0] + trans_x[i]
                #         obj_tmp3['xyz'][1] = obj_tmp2['xyz'][1] + trans_y[i]
                #         obj_tmp3['xyz'][2] = obj_tmp2['xyz'][2] + trans_z[i]
                #         x[i] = np.abs(obj_tmp3['xyz'][0] - obj['xyz'][0])
                #         y[i] = np.abs(obj_tmp3['xyz'][1] - obj['xyz'][1])
                #         z[i] = np.abs(obj_tmp3['xyz'][2] - obj['xyz'][2])
                #         iou[i] = compute_iou3d([l,h,w], [x[i],y[i],z[i]])
                #         # compare the 3d bounding rect and 2d rect
                #         sam_2D = project3d2d(Corners, R, obj_tmp3, Rt)
                #         d3xmin = int(sam_2D[0, :].min())
                #         d3ymin = int(sam_2D[1, :].min())
                #         d3xmax = int(sam_2D[0, :].max())
                #         d3ymax = int(sam_2D[1, :].max())
                #         # # add contour
                #         # d3ymin = d3ymin - int((d3ymax - d3ymin) * contour /2)
                #         # d3ymax = d3ymax + int((d3ymax - d3ymin) * contour /2)
                #         # d3xmin = d3xmin - int((d3xmax - d3xmin) * contour /2)
                #         # d3xmax = d3xmax + int((d3xmax - d3xmin) * contour /2)
                #         if d3ymin < 0: d3ymin = 0
                #         if d3xmin < 0: d3xmin = 0
                #         if d3xmax <= 0: d3xmax = 1
                #         if d3ymax <= 0: d3ymax = 1
                #         if d3ymin >= img.shape[0]-1: d3ymin = img.shape[0] - 2
                #         if d3xmin >= img.shape[1]-1: d3xmin = img.shape[1] - 2
                #         if d3ymax > img.shape[0]-1: d3ymax = img.shape[0] - 1
                #         if d3xmax > img.shape[1]-1: d3xmax = img.shape[1] - 1
                #         Box3Dto2D = [d3xmin, d3ymin, d3xmax, d3ymax]
                #         IoU = compute_iou(Box2D, Box3Dto2D)
                #         if IoU <= topIoU2/2 :
                #             trans_x[i] = trans_fx * l * (np.random.rand(1, 1) * 2 - 1)
                #             trans_y[i] = trans_fy * h * (np.random.rand(1, 1) * 2 - 1)
                #             trans_z[i] = trans_fx * w * (np.random.rand(1, 1) * 2 - 1)
                #         elif IoU > topIoU3 :
                #             topIoU3 = IoU
                #             topIoU_i3 = i
                #
                # obj_tmp3 = copy.deepcopy(obj)
                # obj_tmp3['xyz'][0] = obj_tmp2['xyz'][0] + trans_x[topIoU_i3]
                # obj_tmp3['xyz'][1] = obj_tmp2['xyz'][1] + trans_y[topIoU_i3]
                # obj_tmp3['xyz'][2] = obj_tmp2['xyz'][2] + trans_z[topIoU_i3]
                # x = np.abs(obj_tmp3['xyz'][0] - obj['xyz'][0])
                # y = np.abs(obj_tmp3['xyz'][1] - obj['xyz'][1])
                # z = np.abs(obj_tmp3['xyz'][2] - obj['xyz'][2])
                # print 'xyz:', x, y, z, 'lhw:', l, h, w, 'x/l y/h z/w:', x/l , y/h, z/w,
                # iou3d3 = compute_iou3d([l, h, w], [x, y, z])
                # print 'topIoU3', topIoU3, 'iou3d3', iou3d3
                #
                # n_can = 1024
                # # samples corners
                # trans_fx = 1
                # trans_fy = 0.25
                # trans_fz = 2
                # # sample_type = 'gaussian'
                # # if sample_type == 'gaussian':
                # #     trans = trans_f * np.mean([w, l]) * np.clip(0.5 * np.random.randn(n_can, 2), -1, 1)
                # sample_type = 'uniform'
                # if sample_type == 'uniform':
                #     trans_x = np.insert(trans_fx * l * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_y = np.insert(trans_fy * h * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                #     trans_z = np.insert(trans_fz * w * (np.random.rand(n_can-1, 1) * 2 - 1), 0, values=0, axis=0)
                # x = np.zeros(n_can)
                # y = np.zeros(n_can)
                # z = np.zeros(n_can)
                # iou = np.zeros(n_can)
                # topIoU4 = 0
                # topIoU_i4 = 0
                # for i in range(n_can):
                #     IoU = 0
                #     while IoU <= topIoU3/2 :
                #         obj_tmp4 = copy.deepcopy(obj_tmp)
                #         obj_tmp4['xyz'][0] = obj_tmp3['xyz'][0] + trans_x[i]
                #         obj_tmp4['xyz'][1] = obj_tmp3['xyz'][1] + trans_y[i]
                #         obj_tmp4['xyz'][2] = obj_tmp3['xyz'][2] + trans_z[i]
                #         x[i] = np.abs(obj_tmp4['xyz'][0] - obj['xyz'][0])
                #         y[i] = np.abs(obj_tmp4['xyz'][1] - obj['xyz'][1])
                #         z[i] = np.abs(obj_tmp4['xyz'][2] - obj['xyz'][2])
                #         iou[i] = compute_iou3d([l,h,w], [x[i],y[i],z[i]])
                #         # compare the 3d bounding rect and 2d rect
                #         sam_2D = project3d2d(Corners, R, obj_tmp4, Rt)
                #         d3xmin = int(sam_2D[0, :].min())
                #         d3ymin = int(sam_2D[1, :].min())
                #         d3xmax = int(sam_2D[0, :].max())
                #         d3ymax = int(sam_2D[1, :].max())
                #         # # add contour
                #         # d3ymin = d3ymin - int((d3ymax - d3ymin) * contour /2)
                #         # d3ymax = d3ymax + int((d3ymax - d3ymin) * contour /2)
                #         # d3xmin = d3xmin - int((d3xmax - d3xmin) * contour /2)
                #         # d3xmax = d3xmax + int((d3xmax - d3xmin) * contour /2)
                #         if d3ymin < 0: d3ymin = 0
                #         if d3xmin < 0: d3xmin = 0
                #         if d3xmax <= 0: d3xmax = 1
                #         if d3ymax <= 0: d3ymax = 1
                #         if d3ymin >= img.shape[0]-1: d3ymin = img.shape[0] - 2
                #         if d3xmin >= img.shape[1]-1: d3xmin = img.shape[1] - 2
                #         if d3ymax > img.shape[0]-1: d3ymax = img.shape[0] - 1
                #         if d3xmax > img.shape[1]-1: d3xmax = img.shape[1] - 1
                #         Box3Dto2D = [d3xmin, d3ymin, d3xmax, d3ymax]
                #         IoU = compute_iou(Box2D, Box3Dto2D)
                #         if IoU <= topIoU3/2 :
                #             trans_x[i] = trans_fx * l * (np.random.rand(1, 1) * 2 - 1)
                #             trans_y[i] = trans_fy * h * (np.random.rand(1, 1) * 2 - 1)
                #             trans_z[i] = trans_fx * w * (np.random.rand(1, 1) * 2 - 1)
                #         elif IoU > topIoU4 :
                #             topIoU4 = IoU
                #             topIoU_i4 = i
                #
                # obj_tmp4 = copy.deepcopy(obj)
                # obj_tmp4['xyz'][0] = obj_tmp3['xyz'][0] + trans_x[topIoU_i4]
                # obj_tmp4['xyz'][1] = obj_tmp3['xyz'][1] + trans_y[topIoU_i4]
                # obj_tmp4['xyz'][2] = obj_tmp3['xyz'][2] + trans_z[topIoU_i4]
                # x = np.abs(obj_tmp4['xyz'][0] - obj['xyz'][0])
                # y = np.abs(obj_tmp4['xyz'][1] - obj['xyz'][1])
                # z = np.abs(obj_tmp4['xyz'][2] - obj['xyz'][2])
                # print 'xyz:', x, y, z, 'lhw:', l, h, w, 'x/l y/h z/w:', x/l , y/h, z/w,
                # iou3d4 = compute_iou3d([l, h, w], [x, y, z])
                # print 'topIoU4', topIoU4, 'iou3d4', iou3d4

                # if topIoU4 == 1:
                #     sam_2D = project3d2d(Corners, R, obj_tmp4, Rt)
                #     img_tmp = draw3d(sam_2D, img, (0, 0, 255))
                #     cv2.imshow('image', img_tmp)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()



                line[11] = obj_tmp['xyz'][0]  # [-100,100] left to right
                line[12] = obj_tmp['xyz'][1]  # ground prior
                line[13] = obj_tmp['xyz'][2]  # [1,201]  front
                line = ' '.join([str(item) for item in line]) + ' ' + str(1) + '\n'
                box3d.write(line)