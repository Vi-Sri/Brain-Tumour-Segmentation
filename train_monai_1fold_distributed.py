import argparse
import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.apps import DecathlonDataset
from monai.data import ThreadDataLoader, partition_dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet
from monai.optimizers import Novograd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import sys
sys.path.append('.')
from monai.utils import set_determinism
from brat_tools import *


set_determinism(seed=0)

def main_worker(args):
    # disable logging for processes except 0 on every node
    if int(os.environ["LOCAL_RANK"]) != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"missing directory {args.dir}")

    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    total_start = time.time()
    train_transforms = get_transforms(mode="train")

    train_ds = BratsCacheDataset(
        root_dir=args.dir,
        transform=train_transforms,
        section="training",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=True,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.batch_size, shuffle=True)
    val_transforms = get_transforms(mode="val")
    val_ds = BratsCacheDataset(
        root_dir=args.dir,
        transform=val_transforms,
        section="validation",
        num_workers=4,
        cache_rate=args.cache_rate,
        shuffle=False,
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=args.batch_size, shuffle=False)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceFocalLoss(
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
        batch=True,
    )
    optimizer = Novograd(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model = DistributedDataParallel(model, device_ids=[device])

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    best_metric = -1
    best_metric_epoch = -1
    print(f"time elapsed before training: {time.time() - total_start}")
    train_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        epoch_loss = train(train_loader, model, loss_function, optimizer, lr_scheduler, scaler)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            metric, metric_tc, metric_wt, metric_et = evaluate(
                model, val_loader, dice_metric, dice_metric_batch, post_trans
            )

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), "best_metric_model.pth")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
        f" total train time: {(time.time() - train_start):.4f}"
    )
    dist.destroy_process_group()

def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler):
    model.train()
    step = 0
    epoch_len = len(train_loader)
    epoch_loss = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            images = batch_data["image"].cuda()
            labels = batch_data["label"].cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    lr_scheduler.step()
    epoch_loss /= step

    return epoch_loss

def evaluate(model, val_loader, dice_metric, dice_metric_batch, post_trans):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    inputs=val_data["image"], roi_size=(240, 240, 160), sw_batch_size=4, predictor=model, overlap=0.6
                )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_data["label"])
            dice_metric_batch(y_pred=val_outputs, y=val_data["label"])

        metric = dice_metric.aggregate().item()
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
        dice_metric.reset()
        dice_metric_batch.reset()

    return metric, metric_tc, metric_wt, metric_et

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./dataset", type=str, help="directory of Brain Tumor dataset")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="mini-batch size of every GPU")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="larger cache rate relies on enough GPU memory.")
    parser.add_argument("--val_interval", type=int, default=5)
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)

    main_worker(args=args)

if __name__ == "__main__":
    main()