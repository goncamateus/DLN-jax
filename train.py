import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint
import wandb

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


def parse_args():
    parser = argparse.ArgumentParser(description="DLN-JAX training script")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="random seed to use. Default=123",
    )
    parser.add_argument(
        "--output", default="./output/", help="Location to save checkpoint models"
    )
    parser.add_argument(
        "--model-folder",
        default="DLN-MODEL",
        help="pretrained base model to load",
    )
    parser.add_argument(
        "--fine-tune",
        type=bool,
        default=False,
        help="fine-tune the model with LOL dataset",
    )

    args = parser.parse_args()
    return args


def main(seed, output_folder, fine_tune, model_folder):
    seed = int(time.time())
    name = f"JAX-{seed}-{'LOL' if fine_tune else 'VOC'}"
    wandb.init(
        project="DLN",
        name=name,
        entity="goncamateus",
        config={"seed": seed},
        save_code=True,
    )

    abs_folder_path = os.path.dirname(os.path.abspath(__file__))
    dln_chkpts = f"{abs_folder_path}/models/DLN-MODEL-{seed}/"

    learning_rate = 1e-3
    num_epochs = 2 if fine_tune else 100
    batch_size = 12

    train_state = create_train_state(jax.random.PRNGKey(seed), learning_rate)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if fine_tune:
        indexes_in_folder = sorted([int(i) for i in os.listdir(model_folder)])
        model_folder = f"{model_folder}/{indexes_in_folder[-1]}/default"
        chkpt = orbax_checkpointer.restore(model_folder)
        model_dict = chkpt["model"]
        train_state = train_state.replace(params=model_dict["params"])

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        dln_chkpts, orbax_checkpointer, options
    )
    ckpt = {"model": train_state}
    save_args = orbax_utils.save_args_from_target(ckpt)
    data_loader = LOLLoader if fine_tune else VOC2007Loader
    train_loader = data_loader(
        patch_size=128,
        upscale_factor=1,
        data_augmentation=True,
        batch_size=batch_size,
    )

    first_ll, first_nl = next(train_loader.get(shuffle=False))

    print("Before training:")
    plot_pred(
        train_state, first_ll, first_nl, name=f"{output_folder}/before_training.png"
    )
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
        test_loader = data_loader(
            patch_size=128,
            upscale_factor=1,
            data_augmentation=True,
            batch_size=batch_size,
            train=False,
        )
        test_metrics = evaluate_model(test_state, test_loader)
        for metric, value in test_metrics.items():
            metrics_history[f"test_{metric}"].append(value)

        log_dict = {f"{k}": v[-1] for k, v in metrics_history.items()}
        wandb.log(log_dict)
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
        if epoch % 10 == 0:
            ckpt = {"model": train_state}
            if checkpoint_manager.save(
                epoch, ckpt, save_kwargs={"save_args": save_args}
            ):
                print(f"Saved checkpoint for epoch {epoch}")
                artifact = wandb.Artifact(
                    f"epoch-{epoch}",
                    type="model",
                    metadata={
                        "loss": metrics_history["test_loss"][-1],
                        "PSNR": metrics_history["test_psnr"][-1],
                        "SSIM": metrics_history["test_ssim"][-1],
                    },
                )
                artifact.add_dir(f"{dln_chkpts}/{epoch}")
                wandb.run.log_artifact(artifact)
        end_time = time.time()
        print("Time taken for epoch: ", (end_time - epoch_time_init), "seconds")

    print("After training:")
    plot_pred(
        train_state, first_ll, first_nl, name=f"{output_folder}/after_training.png"
    )
    wandb.log({"Result Image": wandb.Image(f"{output_folder}/after_training.png")})


if __name__ == "__main__":
    args = parse_args()
    main(args.seed, args.output, args.fine_tune, args.model_folder)
