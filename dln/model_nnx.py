import jax
import jax.numpy as jnp

from flax import nnx
from flax import linen as nn


class ConvBlock(nnx.Module):
    def __init__(
        self,
        rngs,
        in_features,
        output_size,
        kernel_size,
        stride,
        padding,
        use_bias=False,
        use_bn=False,
    ):
        self.use_bn = use_bn
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=output_size,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
            # kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
        )

    def __call__(self, x, training: bool = True):
        def prelu(inputs):
            return jnp.where(
                inputs >= 0, inputs, jnp.asarray(0.01, inputs.dtype) * inputs
            )

        x = self.conv(x)
        if self.use_bn:
            x = nnx.BatchNorm(use_running_average=not training)(x)
        x = prelu(x)
        return x


class FA(nnx.Module):

    in_channel: int
    out_channel: int
    reduction: int

    def __init__(self, rngs, in_channel, out_channel, reduction):
        self.linear_1 = nnx.Linear(
            in_channel, in_channel // reduction, use_bias=False, rngs=rngs
        )
        self.linear_2 = nnx.Linear(
            in_channel // reduction, in_channel, use_bias=False, rngs=rngs
        )
        self.conv_block = ConvBlock(
            rngs, in_channel, out_channel, 1, 1, 0, use_bias=True
        )

    def __call__(self, x):
        b, w, h, c = x.shape
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)
        squeezed = jnp.reshape(avg_pool, (b, c))
        double_fc = self.linear_1(squeezed)
        double_fc = nnx.relu(double_fc)
        double_fc = self.linear_2(double_fc)
        double_fc = nnx.sigmoid(double_fc)
        to_expand = jnp.reshape(double_fc, (b, 1, 1, c))
        expand = jnp.broadcast_to(to_expand, (b, w, h, c))
        weights = x * expand
        recalibration = x + weights
        digesting = self.conv_block(recalibration)
        return digesting


class LightenBlock(nnx.Module):
    def __init__(self, rngs, input_size, output_size, kernel_size, stride, padding):
        super(LightenBlock, self).__init__()
        codedim = output_size // 2
        self.encoder = ConvBlock(
            rngs,
            input_size,
            codedim,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )
        self.offset = ConvBlock(
            rngs,
            codedim,
            codedim,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )
        self.normal_light = ConvBlock(
            rngs,
            codedim,
            output_size,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )

    def __call__(self, x):
        encode = self.encoder(x)
        offset = self.offset(encode)
        residual_add = encode + offset
        return self.normal_light(residual_add)


class DarkenBlock(nnx.Module):
    def __init__(
        self,
        rngs,
        input_size,
        output_size,
        kernel_size,
        stride,
        padding,
    ):
        super(DarkenBlock, self).__init__()
        codedim = output_size // 2
        self.encoder = ConvBlock(
            rngs,
            input_size,
            codedim,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )
        self.offset = ConvBlock(
            rngs,
            codedim,
            codedim,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )
        self.low_light = ConvBlock(
            rngs,
            codedim,
            output_size,
            kernel_size,
            stride,
            padding,
            use_bn=False,
            use_bias=True,
        )

    def __call__(self, x):
        encode = self.encoder(x)
        offset = self.offset(encode)
        residual_sub = encode - offset
        return self.low_light(residual_sub)


class LBP(nnx.Module):
    def __init__(self, rngs, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FA(rngs, input_size, output_size, 16)
        self.lighten_1 = LightenBlock(
            rngs, output_size, output_size, kernel_size, stride, padding
        )
        self.darken = DarkenBlock(
            rngs, output_size, output_size, kernel_size, stride, padding
        )
        self.lambda_1 = ConvBlock(
            rngs, output_size, output_size, (1, 1), (1, 1), 0, use_bias=True
        )
        self.lambda_2 = ConvBlock(
            rngs, output_size, output_size, (1, 1), (1, 1), 0, use_bias=True
        )
        self.lighten_2 = LightenBlock(
            rngs, output_size, output_size, kernel_size, stride, padding
        )

    def __call__(self, x):
        x = self.fusion(x)
        nl_est = self.lighten_1(x)
        ll_est = self.darken(nl_est)
        lambda_1 = self.lambda_1(x)
        lambda_2 = self.lambda_2(nl_est)
        residue = ll_est - lambda_1
        nl_est_2 = self.lighten_2(residue)
        y_est = lambda_2 + nl_est_2
        return y_est


class DLN(nnx.Module):

    def __init__(self, rngs, input_dim, dim):
        in_net_dim = input_dim + 1

        self.feat1 = ConvBlock(rngs, in_net_dim, 2 * dim, (3, 3), (1, 1), (1, 1))
        self.feat2 = ConvBlock(rngs, 2 * dim, dim, (3, 3), (1, 1), (1, 1))
        self.lbp_1 = LBP(rngs, dim, dim, (3, 3), (1, 1), (1, 1))
        self.lbp_2 = LBP(rngs, 2 * dim, dim, (3, 3), (1, 1), (1, 1))
        self.lbp_3 = LBP(rngs, 3 * dim, dim, (3, 3), (1, 1), (1, 1))
        self.residual = ConvBlock(rngs, 4 * dim, dim, (3, 3), (1, 1), (1, 1))
        self.out = nnx.Conv(
            rngs=rngs,
            in_features=dim,
            out_features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            # kernel_init=nnx.initializers.kaiming_normal(),
        )

    def __call__(self, x, training: bool = True):
        x = (x - 0.5) * 2
        x_bright = jnp.max(x, axis=3, keepdims=True)
        x_in = jnp.concatenate((x, x_bright), axis=3)
        feat1 = self.feat1(x_in, training=training)
        features = self.feat2(feat1, training=training)
        lbp_1 = self.lbp_1(features)
        lbp2_input = jnp.concatenate((features, lbp_1), axis=3)
        lbp2 = self.lbp_2(lbp2_input)
        lbp3_input = jnp.concatenate((features, lbp_1, lbp2), axis=3)
        lbp3 = self.lbp_3(lbp3_input)
        all_concat = jnp.concatenate((features, lbp_1, lbp2, lbp3), axis=3)
        residual = self.residual(all_concat, training=training)
        to_sum = self.out(residual)
        nl_pred = x + to_sum
        return nl_pred


if __name__ == "__main__":
    model = DLN(nnx.rngs(0), input_dim=3, dim=64)
    y = model(jnp.ones((1, 128, 128, 3)))
    print(y.shape)
