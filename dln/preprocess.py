import tensorflow as tf
import numpy as np
import random


def rescale_img(img, scale):
    img = tf.image.resize(
        img,
        [int(img.shape[0] * scale), int(img.shape[1] * scale)],
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return img


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    ih, iw, _ = img_in.shape
    tp = scale * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    tx, ty = ix * scale, iy * scale

    img_in = img_in[iy : iy + ip, ix : ix + ip]
    img_tar = img_tar[ty : ty + tp, tx : tx + tp]

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    if flip_h and random.random() < 0.5:
        img_in = tf.image.flip_left_right(img_in)
        img_tar = tf.image.flip_left_right(img_tar)

    if rot:
        if random.random() < 0.5:
            img_in = tf.image.flip_up_down(img_in)
            img_tar = tf.image.flip_up_down(img_tar)
        if random.random() < 0.5:
            img_in = tf.image.rot90(img_in, k=2)
            img_tar = tf.image.rot90(img_tar, k=2)

    return img_in, img_tar
