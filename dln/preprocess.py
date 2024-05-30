import tensorflow as tf
import random
from PIL import Image, ImageOps


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img


def rescale_img(img, scale):
    img = tf.image.resize(
        img,
        [int(img.shape[0] * scale), int(img.shape[1] * scale)],
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return img


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((ty, tx, ty + tp, tx + tp))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {"flip_h": False, "flip_v": False, "trans": False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug["flip_h"] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug["flip_v"] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug["trans"] = True

    return img_in, img_tar, info_aug
