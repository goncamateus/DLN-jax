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
            # kernel_init=nn.initializers.kaiming_normal(), <--- DA UM RESULTADO HORRIVEL
            bias_init=nn.initializers.constant(0.0) if self.use_bias else None,
        )(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.PReLU()(x)
        return x


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


class LightenBlock(nn.Module):

    output_size: int
    kernel_size: int
    stride: int
    padding: int
    use_bias: bool

    @nn.compact
    def __call__(self, low_light):
        code_dim = self.output_size // 2
        encode = ConvBlock(code_dim, 3, 1, 1, use_bn=False, use_bias=self.use_bias)(
            low_light
        )
        offset = ConvBlock(code_dim, 3, 1, 1, use_bn=False, use_bias=self.use_bias)(
            encode
        )
        residual_add = encode + offset
        normal_light = ConvBlock(
            self.output_size, 3, 1, 1, use_bn=False, use_bias=self.use_bias
        )(residual_add)
        return normal_light


class DarkenBlock(nn.Module):

    output_size: int
    kernel_size: int
    stride: int
    padding: int
    use_bias: bool

    @nn.compact
    def __call__(self, normal_light):
        code_dim = self.output_size // 2
        encode = ConvBlock(code_dim, 3, 1, 1, use_bn=False, use_bias=self.use_bias)(
            normal_light
        )
        offset = ConvBlock(code_dim, 3, 1, 1, use_bn=False, use_bias=self.use_bias)(
            encode
        )
        residual_sub = encode - offset
        low_light = ConvBlock(
            self.output_size, 3, 1, 1, use_bn=False, use_bias=self.use_bias
        )(residual_sub)
        return low_light


class LBP(nn.Module):

    input_size: int
    output_size: int
    kernel_size: int
    stride: int
    padding: int

    @nn.compact
    def __call__(self, x):
        fusion = FA(self.input_size, self.output_size, 16)(x)
        nl_est = LightenBlock(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            use_bias=True,
        )(fusion)
        ll_est = DarkenBlock(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            use_bias=True,
        )(nl_est)
        lambda_1 = ConvBlock(self.output_size, 1, 1, 0, use_bias=True)(fusion)
        lambda_2 = ConvBlock(self.output_size, 1, 1, 0, use_bias=True)(nl_est)
        nl_est_2 = LightenBlock(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            use_bias=True,
        )(
            ll_est - lambda_1
        )  # Original code was lambda_1 - ll_est but paper is this way i wrote
        y_est = lambda_2 + nl_est_2
        return y_est


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

        lbp_1 = LBP(self.dim, self.dim, 3, 1, 1)(feat2)

        lbp2_input = jnp.concatenate((feat2, lbp_1), axis=3)
        lbp2 = LBP(2 * self.dim, self.dim, 3, 1, 1)(lbp2_input)

        lbp3_input = jnp.concatenate((feat2, lbp_1, lbp2), axis=3)
        lbp3 = LBP(3 * self.dim, self.dim, 3, 1, 1)(lbp3_input)

        all_concat = jnp.concatenate((feat2, lbp_1, lbp2, lbp3), axis=3)
        residual_1 = ConvBlock(4 * self.dim, 3, 1, 1)(all_concat, training=training)
        residual_2 = nn.Conv(
            features=3,
            kernel_size=3,
            strides=1,
            padding=1,
            # kernel_init=nn.initializers.kaiming_normal(), <--- DA UM RESULTADO HORRIVEL
        )(residual_1)
        nl_pred = inputs + residual_2
        return nl_pred


if __name__ == "__main__":
    model = DLN(dim=64)
    sample_input = jnp.ones((32, 128, 128, 3))
    variables = model.init(jax.random.PRNGKey(0), sample_input)
    sample_output = model.apply(variables, sample_input)
    print(sample_output.shape)
