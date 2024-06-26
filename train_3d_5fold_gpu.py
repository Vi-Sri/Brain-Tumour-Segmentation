import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset, CrossValidation
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
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
from monai.utils import set_determinism
from abc import ABC, abstractmethod
import torch
import sys
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

train_transform = Compose(
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
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
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

full_dataset = DecathlonDataset(
    root_dir="./dataset",
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"Fold {fold}")
    train_dataset = Subset(full_dataset, train_ids)
    val_dataset = Subset(full_dataset, val_ids)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=1, shuffle=False)

    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).cuda()
    loss_function = DiceLoss(
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
        batch=True,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    val_dice_metric = DiceMetric(include_background=False, reduction="mean")
    val_hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    max_epochs = 5
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        epoch_loss = 0.0
        dice_metric.reset()
        hausdorff_metric.reset()
        val_dice_metric.reset()
        val_hausdorff_metric.reset()

        for batch_data in tqdm(train_loader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # compute metrics
            dice_metric(y_pred=outputs, y=labels)
            hausdorff_metric(y_pred=outputs, y=labels)
            
        epoch_loss /= len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f} | Train Dice: {dice_metric.aggregate().item():.4f} | Train Hausdorff: {hausdorff_metric.aggregate().item():.4f}")

        # compute metrics on validation set
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                val_dice_metric(y_pred=val_outputs, y=val_labels)
                val_hausdorff_metric(y_pred=val_outputs, y=val_labels)

        print(f"Val Dice: {val_dice_metric.aggregate().item():.4f} | Val Hausdorff: {val_hausdorff_metric.aggregate().item():.4f}")

    print("Training complete.")






