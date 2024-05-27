import numpy as np
import jax

import jax.numpy as jnp

from flax import linen as nn


class ConvBlock(nn.Module):
    output_size: int
    kernel_size: tuple
    stride: tuple
    padding: str  # 'SAME' or 'VALID'
    use_bias: bool = True
    use_bn: bool = False

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(
            features=self.output_size,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
        )(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.PReLU()(x)
        return x


class DLN(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs, training: bool = True):
        x = (inputs - 0.5) * 2
        x_bright = jnp.max(x, axis=3, keepdims=True)
        x_in = jnp.concatenate((x, x_bright), axis=3)
        feat1 = ConvBlock(output_size=2 * self.dim, kernel_size=3, stride=1, padding=1)(
            x_in, training=training
        )
        feat2 = ConvBlock(output_size=self.dim, kernel_size=3, stride=1, padding=1)(
            feat1, training=training
        )
        return feat2


class FA(nn.Module):

    in_channel: int
    out_channel: int
    reduction: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        b, w, h, c = x.shape
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)
        squeezed = jnp.reshape(avg_pool, (b, c))
        double_fc = nn.Sequential(
            [
                nn.Dense(self.in_channel // self.reduction, use_bias=False),
                nn.relu,
                nn.Dense(self.in_channel, use_bias=False),
                nn.sigmoid,
            ]
        )(squeezed)
        to_expand = jnp.reshape(double_fc, (b, 1, 1, c))
        expand = jnp.broadcast_to(to_expand, (b, w, h, c))
        weights = x * expand
        recalibration = x + weights
        digesting = ConvBlock(self.out_channel, 1, 1, 0, use_bias=False)(recalibration)
        return digesting


if __name__ == "__main__":
    model = DLN(dim=64)
    sample_input = jnp.ones((32, 128, 128, 3))
    variables = model.init(jax.random.PRNGKey(0), sample_input)
    sample_output = model.apply(variables, sample_input)
    print(sample_output.shape)
    feature_aggregation = FA(in_channel=64, out_channel=64, reduction=16)
    fa_var = feature_aggregation.init(jax.random.PRNGKey(0), sample_output)
    output = feature_aggregation.apply(fa_var, sample_output)
    print(output.shape)
