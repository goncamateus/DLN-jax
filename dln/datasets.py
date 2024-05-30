import os
import random
import time

import cv2
import tensorflow as tf
import numpy as np

from PIL import Image, ImageEnhance

from dln.preprocess import augment, get_patch, load_img


class VOC2007:
    def __init__(
        self, img_folder, patch_size, upscale_factor, data_augmentation, transform=None
    ):
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

    def __call__(self, index):
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


class LOL:
    def __init__(
        self, img_folder, patch_size, upscale_factor, data_augmentation, transform=None
    ):
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

    def __call__(self, index):
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


def get_dataset(
    dataset_type,
    img_folder,
    patch_size,
    upscale_factor,
    data_augmentation,
    transform=None,
):
    dataset = dataset_type(
        img_folder, patch_size, upscale_factor, data_augmentation, transform
    )
    indices = list(range(len(dataset)))
    tf_dataset = tf.data.Dataset.from_tensor_slices(indices)
    tf_dataset = tf_dataset.map(
        lambda x: tf.numpy_function(dataset, [x], [tf.uint8, tf.uint8]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return tf_dataset


if __name__ == "__main__":
    # Usage example:
    img_folder = "../datasets/train/LOL/low/"
    # Parameters
    patch_size = 128
    upscale_factor = 1
    data_augmentation = True
    batch_size = 32
    num_epochs = 1
    shuffle_buffer_size = 1024
    prefetch_buffer_size = tf.data.AUTOTUNE  # Automatically tune prefetch buffer size
    num_parallel_calls = (
        tf.data.AUTOTUNE
    )  # Automatically tune the number of parallel calls

    # Create the dataset
    tf_dataset = get_dataset(
        LOL, img_folder, patch_size, upscale_factor, data_augmentation
    )

    tf_dataset_parallel = (
        tf_dataset.shuffle(shuffle_buffer_size)
        .repeat(num_epochs)
        .batch(batch_size, drop_remainder=True)
        .map(
            lambda img_in, img_tar: (img_in, img_tar),
            num_parallel_calls=num_parallel_calls,
        )
        .prefetch(prefetch_buffer_size)
    )

    # Measure the time for processing the dataset with parallel loading
    start_time = time.time()
    for img_in, img_tar in tf_dataset_parallel:
        pass
    end_time = time.time()

    print(
        f"Time with parallel loading and prefetching: {end_time - start_time:.2f} seconds"
    )

    # Apply transformations without parallel loading and prefetching
    dataset = tf_dataset.shuffle(shuffle_buffer_size).repeat(num_epochs)
    data_loader = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    # Measure the time for processing the dataset without parallel loading
    start_time = time.time()
    for img_in, img_tar in data_loader:
        pass
    end_time = time.time()

    print(
        f"Time without parallel loading and prefetching: {end_time - start_time:.2f} seconds"
    )
