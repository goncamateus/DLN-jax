import os
import random

import cv2
import numpy as np
import torch.utils.data as data

from PIL import Image, ImageEnhance


from dln.preprocess import augment, get_patch, load_img


class VOC2007(data.Dataset):
    def __init__(
        self, img_folder, patch_size, upscale_factor, data_augmentation, transform=None
    ):
        super(VOC2007, self).__init__()
        self.imgFolder = img_folder
        self.image_filenames = [
            os.path.join(self.imgFolder, x)
            for x in os.listdir(self.imgFolder)
            if self.is_image_file(x)
        ]

        self.image_filenames = self.image_filenames
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def is_image_file(self, filename):
        return any(
            filename.endswith(extension)
            for extension in [".bmp", ".png", ".jpg", ".jpeg"]
        )

    def __getitem__(self, index):

        ori_img = load_img(self.image_filenames[index])  # PIL image
        width, height = ori_img.size
        ratio = min(width, height) / 384

        newWidth = int(width / ratio)
        newHeight = int(height / ratio)
        ori_img = ori_img.resize((newWidth, newHeight), Image.LANCZOS)

        high_image = ori_img

        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7
        ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)
        ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

        ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        ori_img = (ori_img.clip(0, 255)).astype("uint8")
        low_img = ori_img.astype("double") / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        img_in, img_tar = get_patch(
            low_img, high_image, self.patch_size, self.upscale_factor
        )

        if self.data_augmentation:
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)


class LOL(data.Dataset):
    def __init__(
        self, img_folder, patch_size, upscale_factor, data_augmentation, transform=None
    ):
        super(LOL, self).__init__()
        self.img_folder = img_folder
        self.image_filenames = [
            os.path.join(self.img_folder, x)
            for x in os.listdir(self.img_folder)
            if x.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        low_img = load_img(self.image_filenames[index])
        high_image = load_img(self.image_filenames[index].replace("low", "high"))

        img_in, img_tar = get_patch(
            low_img, high_image, self.patch_size, self.upscale_factor
        )

        if self.data_augmentation:
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar
