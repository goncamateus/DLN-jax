import os
import time

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint
import tensorflow as tf

from dm_pix import ssim
from dm_pix import psnr
from flax import nnx
from flax.training import orbax_utils
from tqdm import tqdm

from dln.data_loader import VOC2007Loader, LOLLoader
from dln.model_nnx import DLN
from dln.tv import total_variation


def compute_metrics(*, y_pred, y):
    ssim_loss = (1 - ssim(y, y_pred)).mean()
    tv_loss = total_variation(y_pred, weight=0.001)
    loss = ssim_loss + tv_loss
    res_psnr = jnp.mean(psnr(y, y_pred))
    metrics = {
        "loss": loss,
        "psnr": res_psnr,
        "ssim": 1 - ssim_loss,
    }
    return metrics


def loss_fn(model: DLN, X, y):
    y_pred = model(X)
    ssim_loss = (1 - ssim(y, y_pred)).mean()
    tv_loss = total_variation(y_pred, weight=0.001)
    loss = ssim_loss + tv_loss
    return loss, y_pred


@nnx.jit
def train_step(model: DLN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, X, y):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (_, nl_pred), grads = grad_fn(model, X, y)
    optimizer = optimizer.update(grads)
    res_metrics = compute_metrics(y_pred=nl_pred, y=y)
    metrics.update(
        loss=res_metrics["loss"], psnr=res_metrics["psnr"], ssim=res_metrics["ssim"]
    )


@nnx.jit
def eval_step(model: DLN, metrics: nnx.MultiMetric, X, y):
    nl_pred = model(X)
    res_metrics = compute_metrics(y_pred=nl_pred, y=y)
    metrics.update(
        loss=res_metrics["loss"], psnr=res_metrics["psnr"], ssim=res_metrics["ssim"]
    )


def plot_pred(model, X, y, name="prediction.png"):
    y_pred = model(X)
    print("Before training:")
    print("PSNR:", np.mean(psnr(y, y_pred)))
    print("SSIM:", np.mean(ssim(y, y_pred)))
    plt.subplot(1, 3, 1)
    plt.imshow(X[0])
    plt.title("Low light")
    plt.subplot(1, 3, 2)
    plt.imshow(y[0])
    plt.title("Normal light")
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[0])
    plt.title("Prediction")
    plt.savefig(name)


def main():
    tf.random.set_seed(0)

    seed = int(time.time())
    abs_folder_path = os.path.dirname(os.path.abspath(__file__))
    dln_chkpts = f"{abs_folder_path}/DLN-MODEL-{seed}/"

    learning_rate = 1e-3
    num_epochs = 2
    batch_size = 12

    model = DLN(nnx.Rngs(0), input_dim=3, dim=64)
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        psnr=nnx.metrics.Average("psnr"),
        ssim=nnx.metrics.Average("ssim"),
        loss=nnx.metrics.Average("loss"),
    )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        dln_chkpts, orbax_checkpointer, options
    )
    ckpt = {"model": model}
    save_args = orbax_utils.save_args_from_target(ckpt)

    train_loader = VOC2007Loader(
        patch_size=128,
        upscale_factor=1,
        data_augmentation=True,
        batch_size=batch_size,
    )

    first_ll, first_nl = next(train_loader.get(shuffle=False))

    plot_pred(model, first_ll, first_nl, name="output/before_training.png")
    metrics_history = {
        "train_loss": [],
        "train_psnr": [],
        "train_ssim": [],
        "test_loss": [],
        "test_psnr": [],
        "test_ssim": [],
    }
    for epoch in range(1, num_epochs + 1):
        epoch_time_init = time.time()
        train_steps = len(train_loader) // batch_size
        for X, y in tqdm(train_loader.get(), total=train_steps):
            train_step(model, optimizer, metrics, X, y)

        for metric, value in metrics.compute().items():  # compute metrics
            metrics_history[f"train_{metric}"].append(value)  # record metrics
        metrics.reset()  # reset metrics for test set
        test_loader = LOLLoader(
            patch_size=128,
            upscale_factor=1,
            data_augmentation=False,
            batch_size=batch_size,
            train=False,
        )
        test_steps = len(test_loader) // batch_size
        for X, y in tqdm(test_loader.get(), total=test_steps):
            eval_step(model, metrics, X, y)

        # Log test metrics
        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)
        metrics.reset()  # reset metrics for next training epoch

        print(
            f"train epoch: {epoch}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"PSNR: {metrics_history['train_psnr'][-1]}, "
            f"SSIM: {metrics_history['train_ssim'][-1]}"
        )
        print(
            f"test epoch: {epoch}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"PSNR: {metrics_history['test_psnr'][-1]}, "
            f"SSIM: {metrics_history['test_ssim'][-1]}"
        )
        ckpt = {"model": model}
        if checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args}):
            print(f"Saved checkpoint for epoch {epoch}")
        end_time = time.time()
        print("Time taken for epoch: ", (end_time - epoch_time_init), "seconds")

    plot_pred(model, first_ll, first_nl, name="output/after_training.png")


if __name__ == "__main__":
    main()
