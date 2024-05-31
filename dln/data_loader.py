import numpy as np
import jax.numpy as jnp

from torchvision.transforms import ToTensor
from dln.datasets import LOL, VOC2007


class DataLoader:

    def __init__(
        self,
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
    ):
        self.img_folder = img_folder
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size
        self.setup()

    def setup(self):
        raise NotImplementedError

    def get(self, shuffle=True):
        indices = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(
            0, len(self.dataset) - self.batch_size + 1, self.batch_size
        ):
            excerpt = indices[start_idx : start_idx + self.batch_size]
            batch = [self.dataset[i] for i in excerpt]
            low_light, normal_light = zip(*batch)
            jaxed_low_light = jnp.array(low_light)
            jaxed_normal_light = jnp.array(normal_light)
            yield jnp.transpose(jaxed_low_light, (0, 2, 3, 1)), jnp.transpose(
                jaxed_normal_light, (0, 2, 3, 1)
            )

    def __len__(self):
        return len(self.dataset)


class VOC2007Loader(DataLoader):

    def __init__(
        self,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        train=True,
    ):
        img_folder = f"datasets/{'train' if train else 'test'}/VOC2007/JPEGImages"
        super().__init__(
            img_folder,
            patch_size,
            upscale_factor,
            data_augmentation,
            batch_size,
        )

    def setup(self):
        self.dataset = VOC2007(
            self.img_folder,
            self.patch_size,
            self.upscale_factor,
            self.data_augmentation,
            transform=ToTensor(),
        )


class LOLLoader(DataLoader):

    def __init__(
        self,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        train=True,
    ):
        img_folder = f"datasets/{'train' if train else 'test'}/LOL/low"
        super().__init__(
            img_folder,
            patch_size,
            upscale_factor,
            data_augmentation,
            batch_size,
        )

    def setup(self):
        self.dataset = LOL(
            self.img_folder,
            self.patch_size,
            self.upscale_factor,
            self.data_augmentation,
            transform=ToTensor(),
        )
