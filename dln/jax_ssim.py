import jax
import jax.numpy as jnp
from jax import lax


def gaussian(window_size, sigma):
    x = jnp.arange(window_size)
    gauss = jnp.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
    return gauss / jnp.sum(gauss)


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5)
    _2D_window = _1D_window @ jnp.transpose(_1D_window)
    window = jnp.expand_dims(_2D_window, axis=0)
    reshaped = jnp.broadcast_to(window, (channel, 1, window_size, window_size))


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from skimage.metrics import structural_similarity as skimage_ssim

    def compare_ssim():
        # Create or load two example images
        image1 = (np.random.rand(256, 256).astype(np.float32) * 255).astype(np.uint8)
        image2 = (np.random.rand(256, 256).astype(np.float32) * 255).astype(np.uint8)

        # Compute SSIM using scikit-image
        skimage_ssim_index, _ = skimage_ssim(image1, image2, full=True)
        print("SSIM index (scikit-image):", skimage_ssim_index)

        # Convert images to JAX arrays
        image1_jax = jnp.array(image1)
        image2_jax = jnp.array(image2)

        # Compute SSIM using the handcrafted JAX implementation
        jax_ssim_index = ssim(
            image1_jax, image2_jax
        ).block_until_ready()  # JIT compiled

        print("SSIM index (JAX):", jax_ssim_index)

    compare_ssim()
