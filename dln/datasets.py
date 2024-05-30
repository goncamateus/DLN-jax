import os
import time
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from dln.preprocess import get_patch, augment


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
        ori_img = load_img(self.image_filenames[index])
        ori_img = img_to_array(ori_img)
        width, height = ori_img.shape[1], ori_img.shape[0]
        ratio = min(width, height) / 384
        newWidth, newHeight = int(width / ratio), int(height / ratio)
        ori_img = tf.image.resize(
            ori_img, [newHeight, newWidth], method=tf.image.ResizeMethod.LANCZOS3
        )

        high_image = ori_img

        color_dim_factor = 0.3 * np.random.random() + 0.7
        contrast_dim_factor = 0.3 * np.random.random() + 0.7
        ori_img = tf.image.adjust_saturation(ori_img, color_dim_factor)
        ori_img = tf.image.adjust_contrast(ori_img, contrast_dim_factor)

        low_img = ori_img / 255.0

        beta = 0.5 * np.random.random() + 0.5
        alpha = 0.1 * np.random.random() + 0.9
        gamma = 3.5 * np.random.random() + 1.5
        low_img = beta * tf.pow(alpha * low_img, gamma)
        low_img = low_img * 255.0
        low_img = tf.clip_by_value(low_img, 0, 255)

        img_in, img_tar = get_patch(
            low_img, high_image, self.patch_size, self.upscale_factor
        )

        if self.data_augmentation:
            img_in, img_tar = augment(img_in, img_tar)

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
        low_img = img_to_array(low_img)
        high_image = load_img(self.image_filenames[index].replace("low", "high"))
        high_image = img_to_array(high_image)

        img_in, img_tar = get_patch(
            low_img, high_image, self.patch_size, self.upscale_factor
        )

        if self.data_augmentation:
            img_in, img_tar = augment(img_in, img_tar)

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

    def gen():
        for index in indices:
            img_in, img_tar = dataset(index)
            img_in = tf.image.resize(img_in, [patch_size, patch_size])
            img_tar = tf.image.resize(
                img_tar, [patch_size * upscale_factor, patch_size * upscale_factor]
            )
            yield img_in, img_tar

    tf_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
            tf.TensorSpec(
                shape=(patch_size * upscale_factor, patch_size * upscale_factor, 3),
                dtype=tf.float32,
            ),
        ),
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
