def gen_samples(obj, image_dir, calib_dir):
    n_pos = 512
    pos_regions = np.zeros((n_pos, 107, 107, 3), dtype = 'uint8')
    labels = np.zeros((n_pos, 1), dtype='float')
    image_file = obj['image']
    img = cv2.imread(image_dir + image_file)

    label_file = image_file.replace('png', 'txt')

    # read the camera matrix in the image
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
    # Corners = np.vstack((x_corners, y_corners, z_corners))

    # samples corners
    trans_f = [0.427, 0.151, 0.427]
    n = 1024
    sample_type = 'gaussian'
    flag = 0
    if sample_type == 'gaussian':
        trans = trans_f * np.clip(np.random.randn(n, 3), -10, 10) #- [0.030, -0.035, -0.061]
        
    samples = np.zeros((n,3,8))
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    iou = np.zeros(n)
    for i in range(n):
        samples[i, :, :] = np.vstack((x_corners, y_corners, z_corners))
        x[i] = np.abs(trans[i, 2]*np.cos(-obj['r_y']) + trans[i, 0]*np.sin(-obj['r_y']))
        y[i] = np.abs(trans[i, 1])
        z[i] = np.abs(trans[i, 2]*np.sin(-obj['r_y']) + trans[i, 0]*np.cos(-obj['r_y']))
        iou[i] = compute_iou3d([l,h,w], [x[i],y[i],z[i]])
        # print iou[i]
    # n, bins, patches = plt.hist(iou, 50, density=True, facecolor='g', alpha=0.75)
    # plt.xlabel('iou')
    # plt.ylabel('Probability')
    # plt.title('Histogram of iou')
    # plt.grid(True)
    # plt.show()
    pos_samples = samples
    index = 0
    for i in range(n):
        obj_tmp = copy.deepcopy(obj)
        obj_tmp['xyz'][0] = obj_tmp['xyz'][0] + trans[i, 0]
        obj_tmp['xyz'][1] = obj_tmp['xyz'][1] + trans[i, 1]
        obj_tmp['xyz'][2] = obj_tmp['xyz'][2] + trans[i, 2]
        sam_2D = project3d2d(pos_samples[i, :, :], R, obj_tmp, Rt)
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
        trans_X = np.random.rand(1)
        trans_Y = np.random.rand(1)
        d3ymin = d3ymin - int((d3ymax - d3ymin) * contour * trans_Y)
        d3ymax = d3ymax + int((d3ymax - d3ymin) * contour * (1 - trans_Y))
        d3xmin = d3xmin - int((d3xmax - d3xmin) * contour * trans_X)
        d3xmax = d3xmax + int((d3xmax - d3xmin) * contour * (1 - trans_X))

        if d3ymin < 0: d3ymin = 0
        if d3xmin < 0: d3xmin = 0
        if d3ymax <= int((d3ymax - d3ymin))/2: continue
        if d3xmax <= int((d3xmax - d3xmin))/2: continue
        if d3ymin >= img.shape[0]-int((d3ymax - d3ymin))/2: continue
        if d3xmin >= img.shape[1]-int((d3xmax - d3xmin))/2: continue
        if d3ymax > img.shape[0]: d3ymax = img.shape[0]
        if d3xmax > img.shape[1]: d3xmax = img.shape[1]
        d3patch = img_tmp[d3ymin:d3ymax, d3xmin:d3xmax]  # use layout
        # d3patch = img[d3ymin:d3ymax, d3xmin:d3xmax]  # do not use layout
        try:
            d3patch = imresize(d3patch, (107, 107))
        except:
            print d3ymin,d3ymax,d3xmin,d3xmax
        pos_regions[index] = d3patch
        labels[index] = 2*iou[i]-1
        index = index + 1
        if index == n_pos - 1:
            break

    pos_regions = pos_regions.transpose(0,3,1,2)
    pos_regions = pos_regions.astype('float32') - 128.
    pos_regions = torch.from_numpy(pos_regions).float()
    return pos_regions, labels