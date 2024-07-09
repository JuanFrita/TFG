from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

def add_padding_based_on_stride(img, stride):
    w, h = img.size
    new_h = h
    if h == 1080:
        new_h = 1024
    img_cropped = F.center_crop(img, (w, new_h))
    return img_cropped

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(int(res_h * 0.7), res_h) #avoid reaching non people areas
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        keypoints = np.load(gd_path)
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
           return self.val_transform(img, keypoints)
        elif self.method == 'test':
            return self.test_transform(img, img_path, keypoints)

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
    
    def val_transform(self, img, keypoints):
            wd_original, ht_original = img.size
            
            #remove the necessary padding to the image
            img_padded = add_padding_based_on_stride(img, self.d_ratio)
            wd_padded, ht_padded = img_padded.size
        
            #adjust the keypoints based on the removed padding
            keypoints[:, 0] -= (wd_original - wd_padded) // 2
            keypoints[:, 1] -= (ht_original - ht_padded) // 2
            
            st_size = min(wd_padded, ht_padded)
            
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(0, 0, wd_padded, ht_padded, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            
            return self.trans(img_padded), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(ratio.copy()).float(), st_size, wd_padded, ht_padded
    
    def test_transform(self, img, img_path, keypoints):
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        