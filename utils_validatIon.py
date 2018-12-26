import pickle
import os.path
from os.path import join
import os
import cv2
import copy
import numpy as np
import pickle
import torch
from utils2 import list2file
#10 256  cancel the short <0.1 occlude<0.1
BIN, OVERLAP = 4, 0.1
VEHICLES = ['Car'] # , 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']
# % index for painting
point_idx = np.mat([[1, 3, 6, 8]])
edge_idx = np.mat([[2, 4, 5], [2, 4, 7], [2, 5, 7], [4, 5, 7]])
show_flag = 0
contour = 0.6
n_pos = 64
n_interp = 10
final_image_size = 107
p_size_para = [107, 200]
attri_list = ['train','validation','test']

def project3d2d(Corners, R, obj, Rt):
    # rotate and translate 3D bounding box
    corners_3D = R * Corners  # 3*3 3*8 -> 3*8
    corners_3D[0, :] = corners_3D[0, :] + obj['xyz'][0]
    corners_3D[1, :] = corners_3D[1, :] + obj['xyz'][1]
    corners_3D[2, :] = corners_3D[2, :] + obj['xyz'][2]


    # % project the 3D bounding box into the image plane
    pts_2D = Rt * np.vstack((corners_3D, [1, 1, 1, 1, 1, 1, 1, 1]))  # 4*4 4*8 -> 4*8
    # % scale projected points
    pts_2D[0, :] = np.divide(pts_2D[0, :], pts_2D[2, :])
    pts_2D[1, :] = np.divide(pts_2D[1, :], pts_2D[2, :])

    return pts_2D

def draw3d(pts_2D, img, color, p_size=1, crop_shift = [0,0]):
    # crop shift is the xmin and ymin of crop
    a = crop_shift
    for f in range(4):
        for g in range(3):
            cv2.line(img, (int(pts_2D[0, point_idx[0, f] - 1]) + 1 - a[0], int(pts_2D[1, point_idx[0, f] - 1]) + 1 - a[1]), \
                 (int(pts_2D[0, edge_idx[f, g] - 1]) + 1 - a[0], int(pts_2D[1, edge_idx[f, g] - 1]) + 1 - a[1]), color,
                 p_size, lineType = cv2.CV_AA)
    return img

# write to file the generated images
def get_sample_datas(attri, image_id):
    '''
    attri is in train,validation,test
    id is the 06%d number indicating which photo
    '''
    assert attri in attri_list, "The input attri is wrong!"
    image_dir = join('./dataset', attri, 'image_2')   
    label_dir = join('./dataset', attri, 'label_2')   
    calib_dir = join('./dataset', attri, 'calib')

    f =image_id + ('.png')
    image_file = join(image_dir, f)
    box2d_file = join(label_dir, f.replace('png', 'txt'))
    calib_file = join(calib_dir,f.replace('png', 'txt'))
    
    # get the camera matrix Rt
    for line in open(calib_file).readlines():
        line = line.strip().split(' ')
        if line[0] == 'P2:':
            Rt = np.mat([[float(line[1]),float(line[2]),float(line[3]),float(line[4])], \
                        [float(line[5]),float(line[6]),float(line[7]),float(line[8])], \
                        [float(line[9]),float(line[10]),float(line[11]),float(line[12])],[0,0,0,1]])  # 4*4
    
    # read the image
    img = cv2.imread(image_file)


    # deal with ground truth label for the image

    count_car =  -1
    for line in open(box2d_file).readlines():   # for every object
        line = line.strip().split(' ')
        truncated = np.abs(float(line[1]))
        occluded = np.abs(float(line[2]))

        '''
        delete truncated>0.1 and occluded >1
        '''

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

        '''
        delete ymax - ymin too small
        '''
        y_diff = obj['ymax'] - obj['ymin']
        x_diff = obj['xmax'] - obj['xmin']
        xy_diff = min(x_diff, y_diff)
        if obj['name'] == 'Car' and truncated<0.1 and occluded<=1 and xy_diff >=25:
            # 3D bounding box dimensions
            count_car += 1
            gen_samples(obj,count_car, attri, image_id)
            # print pos_regions.shape
            # write2file(pos_regions, attri, image_id, car_id=0)
            # exit(1)

            # h = obj['real_dims'][0]
            # w = obj['real_dims'][1]
            # l = obj['real_dims'][2]
            # # 3D bounding box corners
            # x_corners = np.mat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
            # y_corners = np.mat([0, 0, 0, 0, -h, -h, -h, -h])
            # z_corners = np.mat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
            # Corners = np.vstack((x_corners, y_corners, z_corners))
            # # compute rotational matrix around yaw axis
            # R = np.mat(
            #     [[np.cos(obj['r_y']), 0, np.sin(obj['r_y'])], [0, 1, 0],
            #      [-np.sin(obj['r_y']), 0, np.cos(obj['r_y'])]])
            # image_2D = project3d2d(Corners, R, obj, Rt)
            # img_tmp = copy.deepcopy(img)
            # # img_tmp = draw3d(image_2D, img_tmp, (0, 255, 0))
            # # extension the cropped image
            # x_diff = int(contour * (obj['xmax'] - obj['xmin']))
            # y_diff = int(contour * (obj['ymax'] - obj['ymin']))
            # xmin_ext = max(0, int(obj['xmin'] - x_diff))
            # ymin_ext = max(0, int(obj['ymin'] - y_diff))
            # xmax_ext = int(obj['xmax'] + x_diff)
            # ymax_ext = int(obj['ymax'] + y_diff)
            # crop_img = img_tmp[ymin_ext : ymax_ext, xmin_ext : xmax_ext]
            # # crop_img = img_tmp[:300,:800]
            # crop_img = draw3d(image_2D, crop_img, (0, 255, 0), crop_shift = [xmin_ext, ymin_ext])
            #print crop_img.shape
              


# input obj and output 256 * 10 images
def gen_samples(obj, car_id, attri, img_id):
    # hyper-parameters
    # pos_regions = np.zeros((n_pos, n_interp, final_image_size, final_image_size, 3), dtype = 'uint8')   # 256 * 10 * 107 * 107 * 3
    # labels = np.zeros((n_pos, n_interp, 3), dtype='float')  #the shift of each image

    #read the images
    image_file = obj['image']
    img = cv2.imread(image_file)
    label_file = image_file.replace('png', 'txt')
    label_file = label_file.replace('image_2','calib')

    # read the camera matrix in the image
    for line in open(label_file).readlines():
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

    # sample the changed x,y,z
    num_sample = 2 * n_pos  # 512

    # samples corners
    trans_f = [0.427, 0.151, 0.427]
    sample_type = 'gaussian'
    if sample_type == 'gaussian':
        # trans.shape = 512 * 3
        trans = trans_f * np.clip(np.random.randn(num_sample, 3), -10, 10) #- [0.030, -0.035, -0.061]


    tmp_labels = np.zeros((n_interp, 3), dtype='float')  #the shift of each image
    count1 = 0
    for i in range(num_sample): # a Gaussian proposal
        obj_most = copy.deepcopy(obj)
        obj_most['xyz'][0] += trans[i, 0]
        obj_most['xyz'][1] += trans[i, 1]
        obj_most['xyz'][2] += trans[i, 2]
        sam_2D_most = project3d2d(Corners, R, obj_most, Rt)   # modified image
        try:
            d3ymin, d3ymax, d3xmin, d3xmax = get_boudary(sam_2D_most, img.shape)
        except:
            continue
        img_tmp = copy.deepcopy(img)
        d3patch = img_tmp[d3ymin:d3ymax, d3xmin:d3xmax]  # use layout

        x_tmp = d3patch.shape[0]
        y_tmp = d3patch.shape[1]
        if(max(x_tmp,y_tmp)>p_size_para[1]):
            p_size = 3
        elif(max(x_tmp,y_tmp)>p_size_para[0]):
            p_size = 2
        else:
            p_size = 1

        for nth_intep in range(n_interp):   # interpolation
            tmp_labels[nth_intep, : ] = trans[i,:] * float(nth_intep) / float(n_interp - 1)    # store the shift of image and interpolation 
            obj_tmp = copy.deepcopy(obj)

            obj_tmp['xyz'][0] += tmp_labels[nth_intep, 0 ]
            obj_tmp['xyz'][1] += tmp_labels[nth_intep, 1 ]
            obj_tmp['xyz'][2] += tmp_labels[nth_intep, 2 ]
            sam_2D = project3d2d(Corners, R, obj_tmp, Rt)   # modified image
            d3patch_tmp = copy.deepcopy(d3patch)
            d3patch_draw = draw3d(sam_2D, d3patch_tmp, (0, 255, 0), p_size = p_size, crop_shift = [d3xmin,d3ymin])
            try:
                d3patch_draw = cv2.resize(d3patch_draw, (107, 107))
            except:
                print d3ymin,d3ymax,d3xmin,d3xmax
            label_xyz_tuple = (obj_tmp['xyz'][0], obj_tmp['xyz'][1],obj_tmp['xyz'][2])  # new x,y,z coordiates
            write2file(d3patch_draw, attri, img_id, label_xyz_tuple, count1, nth_intep, car_id)

        count1 += 1 
        if count1 == n_pos:
            break
    # print pos_regions.shape # (512, 107, 107, 3)
    # pos_regions = pos_regions.transpose(0,3,1,2)
    # pos_regions = pos_regions.astype('float32') - 128.
    # pos_regions = torch.from_numpy(pos_regions).float()
    # print pos_regions.shape
    # print labels[:5]
    # return pos_regions#, labels

def compute_iou3d(A, B):  #  [l,h,w], [x[i],y[i],z[i]])
    if A[0] <= B[0] or A[1] <= B[1] or A[2] <= B[2]:
        return 0
    SXW = B[0] * A[2] * A[1]   # x * w * h
    SLZ = (A[0] - B[0]) *  B[2] * A[1]  # (l-x) * z * h
    SHY = (A[0] - B[0]) * (A[2] - B[2]) * B[1]  # (l-x) *(w-z) * y
    cross = A[0] * A[1] * A[2] - SXW - SLZ -SHY
    IoU = float(cross)/float(A[0] * A[1] * A[2] + SXW + SLZ + SHY)
    return IoU

# write to file the label and image
def write2file(img, attri, image_id, label_xyz_tuple, i, j, car_id=0):  #i j means the 256 * 10 
    rootPath = join('./dataset/new_dataset', attri)
    image_dir = join(rootPath, 'image_2')
    label_dir = join(rootPath, 'label_2')
    out_image_dir = "%s_%01d_%03d_%01d.png" %(image_id, car_id, i, j)
    out_label_dir = out_image_dir.replace('png', 'txt')
    out_image_dir = join(image_dir, out_image_dir)
    out_label_dir = join(label_dir, out_label_dir)
    print out_image_dir
    cv2.imwrite(out_image_dir, img)
    
    out_label_str = str(label_xyz_tuple[0]) +' '+str(label_xyz_tuple[1])+ ' ' +str(label_xyz_tuple[2])
    with open(out_label_dir,"a+") as afile:
        afile.write("%s\n"%(out_label_str))

def get_boudary(sam_2D, img_shape):
    d3xmin = int(sam_2D[0, :].min())
    d3ymin = int(sam_2D[1, :].min())
    d3xmax = int(sam_2D[0, :].max())
    d3ymax = int(sam_2D[1, :].max())

    # add contour
    trans_X = np.random.rand(1)
    trans_Y = np.random.rand(1)
    d3ymin = d3ymin - int((d3ymax - d3ymin) * contour * trans_Y)
    d3ymax = d3ymax + int((d3ymax - d3ymin) * contour * (1 - trans_Y))
    d3xmin = d3xmin - int((d3xmax - d3xmin) * contour * trans_X)
    d3xmax = d3xmax + int((d3xmax - d3xmin) * contour * (1 - trans_X))

    if d3ymin < 0: d3ymin = 0
    if d3xmin < 0: d3xmin = 0
    if d3ymax <= int((d3ymax - d3ymin))/2: return None
    if d3xmax <= int((d3xmax - d3xmin))/2: return None
    if d3ymin >= img_shape[0]-int((d3ymax - d3ymin))/2: return None
    if d3xmin >= img_shape[1]-int((d3xmax - d3xmin))/2: return None
    if d3ymax > img_shape[0]: d3ymax = img_shape[0]
    if d3xmax > img_shape[1]: d3xmax = img_shape[1]
    return d3ymin, d3ymax, d3xmin, d3xmax


def generate_all_data():
    # train_list = get_attri_list('train')
    # validation_list = get_attri_list('validation')
    for attriItem in attri_list[1:2]:    # only generate training and validation
        all_list = get_attri_list(attriItem)
        for img_id in all_list:
            get_sample_datas(attriItem, img_id)




def get_attri_list(attri):
    with open(attri+'_list.txt','rb') as pfile:
        return pickle.load(pfile)



if __name__ == '__main__':
    # with open("val_list.txt","rb") as pfile:
    #     valid_list = pickle.load(pfile)
    #     # valid_list without .png
    # for val_item in valid_list:
    #     try:
    #         img_tmp, truncated = get_sample_datas("validation", val_item)
    #         if truncated < 0.9:
    #             cv2.imwrite("./valid_crop/" + val_item+ ".png",img_tmp)
    #     except TypeError:
    #         pass    

   #  img_tmp,truncated = draw_3D_box("validation","000057.png")
   #  print truncated

    # crop_img = get_sample_datas("validation","000431")
    # #cv2.imwrite("./my_result_img_tmp.png", img_tmp)
    # cv2.imwrite("./my_result_crop.png", crop_img)
    generate_all_data()





