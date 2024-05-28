import numpy as np
import jax.numpy as jnp


def jnp_data_loader(dataset, batch_size, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(dataset) - batch_size + 1, batch_size):
        excerpt = indices[start_idx : start_idx + batch_size]
        batch = [dataset[i] for i in excerpt]
        low_light, normal_light = zip(*batch)
        jaxed_low_light = jnp.array(low_light)
        jaxed_normal_light = jnp.array(normal_light)
        yield jnp.transpose(jaxed_low_light, (0, 2, 3, 1)), jnp.transpose(
            jaxed_normal_light, (0, 2, 3, 1)
        )
