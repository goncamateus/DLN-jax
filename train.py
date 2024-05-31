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
from tqdm import tqdm

from dln.data_loader import VOC2007Loader, LOLLoader
from dln.model import DLN
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


def train_one_epoch(state, data_loader):
    """Train for 1 epoch on the training set."""
    train_steps = len(data_loader) // data_loader.batch_size
    batch_metrics = []
    for X, y in tqdm(data_loader.get(), total=train_steps):
        state, metrics = train_step(state, X, y)
        batch_metrics.append(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(
        batch_metrics
    )  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np


@jax.jit
def eval_step(state, X, y):
    nl_pred = DLN(dim=64).apply({"params": state.params}, X)
    return compute_metrics(y_pred=nl_pred, y=y)


def evaluate_model(state, data_loader):
    """Evaluate on the validation set."""
    batch_metrics = []
    test_steps = len(data_loader) // data_loader.batch_size
    for X, y in tqdm(data_loader.get(), total=test_steps):
        metrics = eval_step(state, X, y)
        batch_metrics.append(metrics)
    # Aggregate the metrics
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
    params = dln.init(key, jnp.ones((1, 128, 128, 3)))["params"]
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=dln.apply, params=params, tx=optimizer
    )


def plot_pred(state, X, y, name="prediction.png"):
    y_pred = DLN(dim=64).apply({"params": state.params}, X)
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

    train_loader = VOC2007Loader(
        patch_size=128,
        upscale_factor=1,
        data_augmentation=True,
        batch_size=batch_size,
    )

    first_ll, first_nl = next(train_loader.get(shuffle=False))

    plot_pred(train_state, first_ll, first_nl, name="output/before_training.png")
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
        train_state, train_metrics = train_one_epoch(train_state, train_loader)
        for metric, value in train_metrics.items():
            metrics_history[f"train_{metric}"].append(value)

        test_state = train_state
        test_loader = LOLLoader(
            patch_size=128,
            upscale_factor=1,
            data_augmentation=False,
            batch_size=batch_size,
            train=False,
        )
        test_metrics = evaluate_model(test_state, test_loader)
        for metric, value in test_metrics.items():
            metrics_history[f"test_{metric}"].append(value)

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
        if checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args}):
            print(f"Saved checkpoint for epoch {epoch}")
        end_time = time.time()
        print("Time taken for epoch: ", (end_time - epoch_time_init), "seconds")

    plot_pred(train_state, first_ll, first_nl, name="output/after_training.png")


if __name__ == "__main__":
    main()
