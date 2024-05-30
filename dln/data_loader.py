from dln.datasets import get_dataset, LOL, VOC2007


class DataLoader:
    dataset_type = None

    def __init__(
        self,
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        num_epochs,
    ):
        self.img_folder = img_folder
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_loader(self):
        raise NotImplementedError


class VOC2007Loader(DataLoader):
    def get_loader(self):
        return (
            get_dataset(
                VOC2007,
                self.img_folder,
                self.patch_size,
                self.upscale_factor,
                self.data_augmentation,
            )
            .shuffle(1024)
            .repeat(self.num_epochs)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(1)
        )


class LOLLoader(DataLoader):
    def get_loader(self):
        return (
            get_dataset(
                LOL,
                self.img_folder,
                self.patch_size,
                self.upscale_factor,
                self.data_augmentation,
            )
            .shuffle(1024)
            .repeat(self.num_epochs)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(1)
        )


def get_VOC2007_train_loader(
    patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
):
    img_folder = "datasets/train/VOC2007/JPEGImages"
    loader = VOC2007Loader(
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        num_epochs,
    )
    return loader.get_loader()


def get_VOC2007_test_loader(
    patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
):
    img_folder = "datasets/test/VOC2007/JPEGImages"
    loader = VOC2007Loader(
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        num_epochs,
    )
    return loader.get_loader()


def get_LOL_train_loader(
    patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
):
    img_folder = "datasets/train/LOL/low"
    loader = LOLLoader(
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        num_epochs,
    )
    return loader.get_loader()


def get_LOL_test_loader(
    patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
):
    img_folder = "datasets/test/LOL/low"
    loader = LOLLoader(
        img_folder,
        patch_size,
        upscale_factor,
        data_augmentation,
        batch_size,
        num_epochs,
    )
    return loader.get_loader()


# Test data loaders
if __name__ == "__main__":
    # Parameters
    patch_size = 128
    upscale_factor = 1
    data_augmentation = True
    batch_size = 32
    num_epochs = 1

    # Test VOC2007 data loaders
    train_loader = get_VOC2007_train_loader(
        patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
    )
    test_loader = get_VOC2007_test_loader(
        patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
    )

    print("VOC2007")
    print("Train loader")
    for img_in, img_tar in train_loader.take(1):
        print(img_in.shape, img_tar.shape)
    print("Test loader")
    for img_in, img_tar in test_loader.take(1):
        print(img_in.shape, img_tar.shape)

    # Test LOL data loaders
    train_loader = get_LOL_train_loader(
        patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
    )
    test_loader = get_LOL_test_loader(
        patch_size, upscale_factor, data_augmentation, batch_size, num_epochs
    )
    print("LOL")
    print("Train loader")
    for img_in, img_tar in train_loader.take(1):
        print(img_in.shape, img_tar.shape)
    print("Test loader")
    for img_in, img_tar in test_loader.take(1):
        print(img_in.shape, img_tar.shape)
