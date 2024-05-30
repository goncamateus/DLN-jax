import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from dln.data import get_Low_light_training_set
from dln.data import get_Low_light_test_set
from dln.jax_dln import DLN
from dln.jax_data_loader import jnp_data_loader
from dln.jax_tv import total_variation
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


def train_one_epoch(state, train_set, batch_size):
    """Train for 1 epoch on the training set."""
    data_loader = jnp_data_loader(train_set, batch_size=batch_size)
    train_steps = len(train_set) // batch_size
    batch_metrics = []
    for X, y in tqdm(data_loader, total=train_steps):
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


def evaluate_model(state, test_set, batch_size):
    """Evaluate on the validation set."""
    data_loader = jnp_data_loader(test_set, batch_size=batch_size)
    X, y = next(data_loader)
    metrics = eval_step(state, X, y)
    metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    metrics = jax.tree.map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    return metrics


def create_train_state(key, learning_rate):
    dln = DLN(dim=64)
    params = dln.init(key, jnp.ones((1, 256, 256, 3)))["params"]
    optimizer = optax.adam(learning_rate=learning_rate)
    return train_state.TrainState.create(
        apply_fn=dln.apply, params=params, tx=optimizer
    )


def get_datasets():
    train_set = get_Low_light_training_set(
        upscale_factor=1, patch_size=128, data_augmentation=True
    )
    test_set = get_Low_light_test_set(
        upscale_factor=1, patch_size=128, data_augmentation=True
    )

    return train_set, test_set


def plot_pred(params, X, y, name="prediction.png"):
    y_pred = DLN(dim=64).apply({"params": params}, X)
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
    abs_folder_path = os.path.dirname(os.path.abspath(__file__))
    dln_chkpts = f"{abs_folder_path}/DLN-MODEL-{int(time.time())}/"

    seed = 0  # needless to say these should be in a config or defined like flags
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 12

    train_state = create_train_state(jax.random.PRNGKey(seed), learning_rate)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=101, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        dln_chkpts, orbax_checkpointer, options
    )
    ckpt = {"model": train_state}
    save_args = orbax_utils.save_args_from_target(ckpt)

    train_set, test_set = get_datasets()
    data_loader = jnp_data_loader(train_set, batch_size=batch_size)
    first_ll, first_nl = next(data_loader)
    del data_loader

    plot_pred(train_state.params, first_ll, first_nl, name="before_training.png")

    for epoch in range(1, num_epochs + 1):
        train_state, train_metrics = train_one_epoch(train_state, train_set, batch_size)
        print(
            f"Train epoch: {epoch}, loss: {train_metrics['loss']}, psnr: {train_metrics['psnr']}, ssim: {train_metrics['ssim']}"
        )
        test_state = train_state
        test_metrics = evaluate_model(test_state, test_set, batch_size)
        print(
            f"Test epoch: {epoch}, loss: {test_metrics['loss']}, psnr: {test_metrics['psnr']}, ssim: {train_metrics['ssim']}"
        )
        ckpt = {"model": train_state}
        if checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args}):
            print(f"Saved checkpoint for epoch {epoch}")

    plot_pred(train_state.params, first_ll, first_nl, name="after_training.png")


if __name__ == "__main__":
    main()
