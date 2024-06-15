import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint

from PIL import Image

from flax.core import freeze, unfreeze
from torchvision import transforms


from dln.model import DLN


def parse_args():
    parser = argparse.ArgumentParser(description="JAX DLN Inference")
    parser.add_argument(
        "--image-dataset",
        type=str,
        default="datasets/test/LOL/low/",
        help="Path to the input images",
    )
    parser.add_argument(
        "--output", default="./output/", help="Location to save the output images"
    )
    parser.add_argument(
        "--model-folder",
        default="DLN-MODEL",
        help="Model folder to load the model from. Default=DLN-MODEL",
    )

    args = parser.parse_args()
    return args


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"]
    )


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
def eval_step(params, X):
    nl_pred = DLN(dim=64).apply({"params": params}, X)
    return nl_pred


def eval_over_images(LL_images, pred_images, model_params):
    time_ave = 0
    trans = transforms.ToTensor()
    for i in range(len(LL_images)):
        LL_in = Image.open(LL_images[i])
        LL_torch = trans(LL_in).permute(1, 2, 0).unsqueeze(0)
        LL = jnp.array(LL_torch)
        t0 = time.time()
        nl_pred = eval_step(model_params, LL)
        t1 = time.time()
        if i != 0:  # skip the first image count due to gpu loading time
            time_ave += t1 - t0
        nl_pred = nl_pred * 255
        nl_pred = nl_pred.clip(0, 255)
        nl_pred = np.array(nl_pred, dtype=np.uint8).squeeze()
        Image.fromarray(nl_pred).save(pred_images[i])
        print(
            "===> Processing Image: %04d /%04d in %.4f s."
            % (i + 1, len(LL_images), (t1 - t0))
        )
    print("===> Processing Time: %.4f ms." % (time_ave / len(LL_images) * 1000))


def load_model_params(model_folder):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    indexes_in_folder = sorted([int(i) for i in os.listdir(model_folder)])
    model_folder = f"{model_folder}/{indexes_in_folder[-1]}/default"
    chkpt = orbax_checkpointer.restore(model_folder)
    model_params = chkpt["model"]["params"]
    return model_params


def count_params(params):
    # Flatten the nested dictionary to get all parameter arrays
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened_params = flatten_dict(unfreeze(params))
    return sum(jnp.prod(jnp.array(p.shape)) for p in flattened_params.values())


if __name__ == "__main__":
    args = parse_args()
    model_params = load_model_params(args.model_folder)
    num_params = count_params(model_params)
    print(f"Number of parameters in Flax model: {num_params}")

    LL_images, pred_images = image_paths(args.image_dataset, args.output)
    eval_over_images(LL_images, pred_images, model_params)
