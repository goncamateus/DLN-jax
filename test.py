import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


from dln.dataset import is_image_file
from dln.model import DLN

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
    parser.add_argument(
        "--upscale_factor", type=int, default=1, help="super resolution upscale factor"
    )
    parser.add_argument("--chop-forward", type=bool, default=True)
    parser.add_argument(
        "--patch-size", type=int, default=256, help="0 to use original frame size"
    )
    parser.add_argument(
        "--stride", type=int, default=16, help="0 to use original patch size"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="random seed to use. Default=123"
    )
    parser.add_argument("--gpus", default=1, type=int, help="number of gpu")
    parser.add_argument("--image-dataset", type=str, default="test_img")
    parser.add_argument(
        "--output", default="./output/", help="Location to save checkpoint models"
    )
    parser.add_argument(
        "--model-folder",
        default="DLN-MODEL",
        help="sr pretrained base model",
    )
    parser.add_argument(
        "--image-based", type=bool, default=True, help="use image or video based ULN"
    )
    parser.add_argument("--chop", type=bool, default=False)

    args = parser.parse_args()
    return args


def image_paths(data_path, out_path):
    LL_filenames = os.path.join(data_path)
    pred_filenames = os.path.join(out_path)
    try:
        os.stat(pred_filenames)
    except:
        os.mkdir(pred_filenames)
    LL_images = [
        os.path.join(LL_filenames, x)
        for x in sorted(os.listdir(LL_filenames))
        if is_image_file(x)
    ]
    pred_images = [
        os.path.join(pred_filenames, x)
        for x in sorted(os.listdir(LL_filenames))
        if is_image_file(x)
    ]

    return LL_images, pred_images


@jax.jit
def eval_step(state, X):
    nl_pred = DLN(dim=64).apply({"params": state.params}, X)
    return nl_pred


def eval_over_images(LL_images, pred_images, state):
    trans = transforms.ToTensor()
    for i in tqdm(range(len(LL_images))):
        LL_in = Image.open(LL_images[i]).convert("RGB")
        LL_torch = trans(LL_in).permute(1, 2, 0).unsqueeze(0)
        LL = jnp.array(LL_torch)
        print(LL.shape)


if __name__ == "__main__":
    args = parse_args()
    LL_images, pred_images = image_paths(args.image_dataset, args.output)
    eval_over_images(LL_images, pred_images, None)
