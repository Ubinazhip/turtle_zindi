from torch.utils.data.dataset import Dataset
import os
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import torch
from tqdm import tqdm
import pandas as pd

im_size = 680
crop_size = 640

def data_transforms():
    return {
        'train': albumentations.Compose([
            albumentations.Resize(im_size, im_size),
            albumentations.RandomCrop(crop_size, crop_size),
            #albumentations.RandomBrightnessContrast(p = 0.4, brightness_limit=0.25, contrast_limit=0.25),
            albumentations.HorizontalFlip(p=0.5),
            #albumentations.OneOf([
            #    albumentations.CLAHE(),
            #    albumentations.RandomGamma()
            #    ], p=1.0
            #),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.15, shift_limit=0.1, p=0.9),
            albumentations.OneOf([
                albumentations.Blur(blur_limit=7, p=1.0),
                albumentations.MotionBlur(),
                albumentations.GaussNoise(),
                albumentations.ImageCompression(quality_lower=75)
            ], p=0.5),

            albumentations.Cutout(num_holes=12, max_h_size=35, max_w_size=35, p=0.5),
            albumentations.Normalize(),
            ToTensorV2(),
        ]),
        'val': albumentations.Compose([
            albumentations.Resize(im_size, im_size),
            albumentations.CenterCrop(crop_size, crop_size),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
}

#im_size = 1096
#crop_size = 1024

def data_transforms_new():
    return {
        'train': albumentations.Compose([
            albumentations.Resize(im_size, im_size),
            albumentations.RandomCrop(crop_size, crop_size),
            #albumentations.RandomBrightnessContrast(p = 0.4, brightness_limit=0.25, contrast_limit=0.25),
            albumentations.HorizontalFlip(p=0.5),
            #albumentations.OneOf([
            #    albumentations.CLAHE(),
            #    albumentations.RandomGamma()
            #    ], p=1.0
            #),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.15, shift_limit=0.1, p=0.9),
            albumentations.OneOf([
                albumentations.Blur(blur_limit=7, p=1.0),
                albumentations.MotionBlur(),
                albumentations.GaussNoise(),
                albumentations.ImageCompression(quality_lower=75)
            ], p=0.5),

            #albumentations.Cutout(num_holes=12, max_h_size=35, max_w_size=35, p=0.5),
            albumentations.Normalize(),
            ToTensorV2(),
        ]),
        'val': albumentations.Compose([
            albumentations.Resize(im_size, im_size),
            albumentations.CenterCrop(crop_size, crop_size),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
}

def data_transforms2():
    return {
        'train': albumentations.Compose([
            albumentations.Resize(412, 412),
            albumentations.RandomCrop(384, 384),
            #albumentations.RandomBrightnessContrast(p = 0.4, brightness_limit=0.25, contrast_limit=0.25),
            albumentations.HorizontalFlip(p=0.5),
            #albumentations.OneOf([
            #    albumentations.CLAHE(),
            #    albumentations.RandomGamma()
            #    ], p=1.0
            #),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.15, shift_limit=0.1, p=0.9),
            albumentations.OneOf([
                albumentations.Blur(blur_limit=7, p=1.0),
                albumentations.MotionBlur(),
                albumentations.GaussNoise(),
                albumentations.ImageCompression(quality_lower=75)
            ], p=0.5),

            albumentations.Cutout(num_holes=12, max_h_size=10, max_w_size=10, p=0.5),
            albumentations.Normalize(),
            ToTensorV2(),
        ]),
        'val': albumentations.Compose([
            albumentations.Resize(412, 412),
            albumentations.CenterCrop(384, 384),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
}

class CustomDataset(Dataset):
    def __init__(self, transform, split, folder, fold):
        self.transform = transform
        self.split = split
        self.data_folder = "/home/tf/train_data"
        self.csv_folder = "/home/tf/turtle_data"

        #self.data = pd.read_csv(os.path.join(self.csv_folder, f"main_folds/{self.split}_fold{fold}.csv"))
        #self.data = pd.read_csv(os.path.join(self.csv_folder, f"folds/{self.split}_folds{fold}.csv"))
        self.data = pd.read_csv(os.path.join(self.csv_folder, f"{folder}/{self.split}_folds{fold}.csv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_folder, self.data["image_id"][idx] + ".JPG"))
        transformed = self.transform(image=image)
        image = transformed["image"]
        cls = self.data["class"][idx]

        return image, cls
