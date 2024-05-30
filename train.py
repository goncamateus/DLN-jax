import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint

from dm_pix import ssim
from dm_pix import psnr
from flax.training import train_state
from flax.training import orbax_utils

from dln.data_loader import (
    get_VOC2007_train_loader,
    get_VOC2007_test_loader,
    get_LOL_train_loader,
    get_LOL_test_loader,
)
from dln.model import DLN
from dln.tv import total_variation
from tqdm import tqdm


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


@jax.jit
def train_step(state, X, y):
    def loss_fn(params):
        y_pred = DLN(dim=64).apply({"params": params}, X)
        ssim_loss = (1 - ssim(y, y_pred)).mean()
        tv_loss = total_variation(y_pred, weight=0.001)
        loss = ssim_loss + tv_loss
        return loss, y_pred

    (_, nl_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    metrics = compute_metrics(y_pred=nl_pred, y=y)
    return state, metrics


@jax.jit
def eval_step(state, X, y):
    nl_pred = DLN(dim=64).apply({"params": state.params}, X)
    return compute_metrics(y_pred=nl_pred, y=y)


def evaluate_model(state, batch_size):
    """Evaluate on the validation set."""
    test_ds = get_LOL_test_loader(
        patch_size=128,
        upscale_factor=1,
        data_augmentation=True,
        batch_size=batch_size,
        num_epochs=1,
    )
    batch_metrics = []
    for ll, nl in test_ds.as_numpy_iterator():
        X = jnp.float32(ll)
        y = jnp.float32(nl)
        metrics = eval_step(state, X, y)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(
        batch_metrics
    )  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np


def create_train_state(key, learning_rate):
    dln = DLN(dim=64)
    params = dln.init(key, jnp.ones((1, 256, 256, 3)))["params"]
    optimizer = optax.adam(learning_rate=learning_rate)
    return train_state.TrainState.create(
        apply_fn=dln.apply, params=params, tx=optimizer
    )


def plot_pred(params, X, y, name="prediction.png"):
    y_pred = DLN(dim=64).apply({"params": params}, X)
    print("Before training:")
    print("PSNR:", np.mean(psnr(y, y_pred)))
    print("SSIM:", np.mean(ssim(y, y_pred)))
    plt.subplot(1, 3, 1)
    plt.imshow(X[0].astype(jnp.uint8))
    plt.title("Low light")
    plt.subplot(1, 3, 2)
    plt.imshow(y[0].astype(jnp.uint8))
    plt.title("Normal light")
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[0].astype(jnp.uint8))
    plt.title("Prediction")
    plt.savefig(name)


def main():
    seed = int(time.time())
    abs_folder_path = os.path.dirname(os.path.abspath(__file__))
    dln_chkpts = f"{abs_folder_path}/DLN-MODEL-{seed}/"

    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 12

    train_state = create_train_state(jax.random.PRNGKey(seed), learning_rate)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        dln_chkpts, orbax_checkpointer, options
    )
    ckpt = {"model": train_state}
    save_args = orbax_utils.save_args_from_target(ckpt)

    train_loader = get_VOC2007_train_loader(
        patch_size=128,
        upscale_factor=1,
        data_augmentation=True,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    for img_in, img_tar in train_loader.take(1):
        first_ll, first_nl = img_in, img_tar

    first_ll = jnp.float32(first_ll)
    first_nl = jnp.float32(first_nl)
    plot_pred(train_state.params, first_ll, first_nl, name="output/before_training.png")
    num_steps_per_epoch = train_loader.cardinality().numpy() // num_epochs
    print(num_steps_per_epoch)
    metrics_history = {
        "train_loss": [],
        "train_psnr": [],
        "train_ssim": [],
        "test_loss": [],
        "test_psnr": [],
        "test_ssim": [],
    }
    batch_metrics = []

    for step, (ll, nl) in tqdm(enumerate(train_loader.as_numpy_iterator())):
        X = jnp.float32(ll)
        y = jnp.float32(nl)
        train_state, metrics = train_step(train_state, X, y)
        batch_metrics.append(metrics)

        epoch = (step + 1) // num_steps_per_epoch
        if (
            epoch > 0 and (step + 1) % num_steps_per_epoch == 0
        ):  # one training epoch has passed
            # Aggregate the metrics
            batch_metrics_np = jax.device_get(
                batch_metrics
            )  # pull from the accelerator onto host (CPU)
            epoch_metrics_np = {
                k: np.mean([metrics[k] for metrics in batch_metrics_np])
                for k in batch_metrics_np[0]
            }
            metrics_history["train_loss"].append(epoch_metrics_np["loss"])
            metrics_history["train_psnr"].append(epoch_metrics_np["psnr"])
            metrics_history["train_ssim"].append(epoch_metrics_np["ssim"])
            batch_metrics = []

            # Compute metrics on the test set after each training epoch
            eval_metrics = evaluate_model(train_state, batch_size)

            # Log test metrics
            metrics_history["test_loss"].append(eval_metrics["loss"])
            metrics_history["test_psnr"].append(eval_metrics["psnr"])
            metrics_history["test_ssim"].append(eval_metrics["ssim"])

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
            ckpt = {"model": train_state}
            if checkpoint_manager.save(
                epoch, ckpt, save_kwargs={"save_args": save_args}
            ):
                print(f"Saved checkpoint for epoch {epoch}")

    plot_pred(train_state.params, first_ll, first_nl, name="output/after_training.png")


if __name__ == "__main__":
    main()
