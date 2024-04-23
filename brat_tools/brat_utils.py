from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.apps import DecathlonDataset
from monai.data import ThreadDataLoader, partition_dataset, decollate_batch
import torch
import torch.distributed as dist
import os

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
def get_transforms(mode="train"):
    if mode == "train":
        return Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    else:
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

class BratsCacheDataset(DecathlonDataset):
    """
    Enhance the DecathlonDataset to support distributed data parallel.

    """
    def __init__(
        self,
        root_dir,
        section,
        transform=None,
        cache_rate=1.0,
        num_workers=0,
        shuffle=False,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("root directory root_dir must be a directory.")
        self.section = section
        self.shuffle = shuffle
        self.val_frac = 0.2
        self.set_random_state(seed=0)
        dataset_dir = os.path.join(root_dir, "Task01_BrainTumour")
        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"cannot find dataset directory: {dataset_dir}, please download it from Decathlon challenge."
            )
        data = self._generate_data_list(dataset_dir)
        super(DecathlonDataset, self).__init__(data, transform, cache_rate=cache_rate, num_workers=num_workers)

    def _generate_data_list(self, dataset_dir):
        data = super()._generate_data_list(dataset_dir)
        # partition dataset based on current rank number, every rank trains with its own data
        # it can avoid duplicated caching content in each rank, but will not do global shuffle before every epoch
        return partition_dataset(
            data=data,
            num_partitions=dist.get_world_size(),
            shuffle=self.shuffle,
            seed=0,
            drop_last=False,
            even_divisible=self.shuffle,
        )[dist.get_rank()]
    


    
    
