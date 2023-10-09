import os
import numpy as np
import json

# root = '/root/Downloads/ISS/'
root = '/root/Downloads/Jilin-248/'
# root = '/root/Downloads/Skysat-1/'
# json_name = 'ISS.json'
json_name = 'Jilin-248.json'

dataset_dict = {}

seq_list = [x for x in os.listdir(root) if '.txt' not in x]

for seq in seq_list:
    seq_dict = {}
    seq_dict['video_dir'] = seq
    img_list = os.listdir(os.path.join(root, seq, 'img'))
    img_list.sort()
    seq_dict['img_names'] = [os.path.join(seq, 'img', x) for x in img_list]

    with open(os.path.join(root, seq, 'groundtruth_rect.txt'), 'r') as f:
        gt = f.readlines()
    tmp = None
    new_gt = []
    for i in range(len(gt)):
        split = list(map(float, gt[i].split(',')))
        if len(gt[i]) > 4:
            new_gt.append(split)
            tmp = split
        else:
            new_gt.append(tmp)
    gt = new_gt

    seq_dict['init_rect'] = gt[0]
    seq_dict['gt_rect'] = gt

    # attr = np.loadtxt(os.path.join(root, seq, 'attribute.txt'), delimiter=',')
    seq_dict['attr'] = []

    dataset_dict[seq] = seq_dict

with open(os.path.join(root, json_name), 'w') as f:
    json.dump(dataset_dict, f)
