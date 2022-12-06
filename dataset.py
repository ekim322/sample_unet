import os
import glob
from PIL import Image
import csv
import numpy as np

import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, csv_path, img_size=224, transform=None, eval=False):
        # img_paths = glob.glob(os.path.join(data_dir, 'images/*.png'))
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            img_paths = list(reader)
            img_paths = [path[0] for path in img_paths]

        mask_paths = [x.replace('images','masks').replace('image','mask').replace('png','npy') for x in img_paths]
        
        self.img_size = img_size
        self.transform = transform
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.eval = eval
        
    def __len__(self):
        return len(self.img_paths)
    
    def get_naive_bbox(self, img):
        pixels = np.where(img!=0)

        min_row = np.min(pixels[0])
        max_row = np.max(pixels[0])
        min_col = np.min(pixels[1])
        max_col = np.max(pixels[1])
            
        return min_row, max_row, min_col, max_col
    
    def get_crop_idx(self, mask, get_min=False):
        mask_bbox = self.get_naive_bbox(mask)
        bbox_h_shift = int((mask_bbox[1] - mask_bbox[0])*3)
        bbox_w_shift = int((mask_bbox[3] - mask_bbox[2])*3)

        if get_min:
            r = 10
            crop_start_row = max(0, int(mask_bbox[0]-bbox_h_shift/r))
            crop_end_row = min(mask.shape[0], int(mask_bbox[1]+bbox_h_shift/r))
            crop_start_col = max(0, mask_bbox[2]-int(bbox_w_shift/r))
            crop_end_col = min(mask.shape[1], int(mask_bbox[3]+bbox_w_shift/r))
        
        else:
            crop_start_row = max(0, mask_bbox[0]-np.random.randint(bbox_h_shift/20, bbox_h_shift))
            crop_end_row = min(mask.shape[0], mask_bbox[1]+np.random.randint(bbox_h_shift/20, bbox_h_shift))
            crop_start_col = max(0, mask_bbox[2]-np.random.randint(bbox_w_shift/20, bbox_w_shift))
            crop_end_col = min(mask.shape[1], mask_bbox[3]+np.random.randint(bbox_w_shift/20, bbox_w_shift))
        
        return crop_start_row, crop_end_row, crop_start_col, crop_end_col
        
    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.load(self.mask_paths[idx])
        
        r1, r2, c1, c2 = self.get_crop_idx(mask)
        mask = mask[r1:r2, c1:c2]
        img = img[r1:r2, c1:c2, :]
        
        if self.eval:
            r1, r2, c1, c2 = self.get_crop_idx(mask, get_min=True)
            rotate_angle = 0
        else:
            r1, r2, c1, c2 = self.get_crop_idx(mask)
            rotate_angle = np.random.randint(0,45) * (-1 if np.random.choice([True, False]) else 1)

        # Crop ROI
        mask = mask[r1:r2, c1:c2]
        img = img[r1:r2, c1:c2, :]

        img_transform = A.Compose([
            A.Resize(width=self.img_size, height=self.img_size),
            A.Rotate(limit=(rotate_angle,rotate_angle+1), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
        ])

        mask_transform =A.Compose([
            A.Resize(width=self.img_size, height=self.img_size,\
                     interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=(rotate_angle,rotate_angle+1), p=1, \
                     interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        ])
        
        # Augment Images
        img_augmentations = img_transform(image=img)
        mask_augmentations = mask_transform(image=mask)
        img = img_augmentations['image']
        mask = mask_augmentations['image']
        
        img = img / 255
        
        img = torch.from_numpy(np.transpose(img, (2,0,1)))
        mask = torch.from_numpy(mask)
        
        return img, mask