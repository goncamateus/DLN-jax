import jax
import jax.numpy as jnp


# @jax.jit
def total_variation(x, weight=1):
    b, h, w, c = x.shape
    count_h = h * (w - 1) * c
    count_w = (h - 1) * w * c
    h_tv = jnp.sum(jnp.square(x[:, 1:, :, :] - x[:, : h - 1, :, :]))
    w_tv = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, : w - 1, :]))
    return weight * 2 * (h_tv / count_h + w_tv / count_w) / b
