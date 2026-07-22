"""DBSegNet training loop preserving the reported experiment behavior."""

import json
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from nets import DBSegNet
from nets.segformer_training import get_lr_scheduler, set_optimizer_lr, weights_init
from training.checkpoint import load_matching_weights
from training.experiment import make_run_dir, read_split, save_run_manifest
from training.optimizers import build_optimizer, fitted_learning_rates
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import ModelEMA, fit_one_epoch


def run_training(config):
    if config.train.distributed:
        raise NotImplementedError(
            "The reported 81.5% protocol used single-process training."
        )

    seed_everything(config.train.seed)
    cuda = config.train.cuda and torch.cuda.is_available()
    fp16 = config.train.fp16 and cuda
    split_dir = config.data.resolved_split_dir()

    model = DBSegNet(
        num_classes=config.model.num_classes,
        in_channels=config.model.input_channels,
        branch=config.model.branch,
        fusion_mode=config.model.fusion_mode,
        use_cross_attention=config.model.use_cross_attention,
    )
    weights_init(model, init_type=config.model.initialization)
    if config.model.checkpoint:
        print("Loaded checkpoint:", load_matching_weights(model, config.model.checkpoint))

    # Keep this before DataLoader creation: SummaryWriter graph tracing consumes
    # RNG state in the original training process.
    run_dir = make_run_dir(config)
    log_dir = run_dir / "loss"
    loss_history = LossHistory(
        str(log_dir),
        model,
        input_shape=config.data.input_shape,
        input_channels=config.model.input_channels,
    )

    scaler = GradScaler() if fp16 else None
    model_train = model.train()
    if config.train.sync_bn:
        print("SyncBN is unavailable in the single-process reproduction protocol.")
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    model_ema = (
        ModelEMA(model_train, config.train.ema_decay)
        if config.train.use_ema
        else None
    )

    train_lines = read_split(split_dir / "train.txt")
    val_lines = read_split(split_dir / "val.txt")
    num_train, num_val = len(train_lines), len(val_lines)
    if not num_train or not num_val:
        raise ValueError("train.txt and val.txt must both contain samples")

    initial_lr, final_lr = fitted_learning_rates(config.train)
    optimizer = build_optimizer(model, config.train, initial_lr)
    lr_scheduler = get_lr_scheduler(
        config.train.lr_schedule,
        initial_lr,
        final_lr,
        config.train.epochs,
    )

    train_dataset = SegmentationDataset(
        train_lines,
        config.data.input_shape,
        config.model.num_classes,
        True,
        config.data.dataset_path,
        aug_mode=config.data.augmentation,
        dataset_type=config.data.dataset_type,
        small_target_strategy=False,
    )
    val_dataset = SegmentationDataset(
        val_lines,
        config.data.input_shape,
        config.model.num_classes,
        False,
        config.data.dataset_path,
        aug_mode="resize_only",
        dataset_type=config.data.dataset_type,
    )
    loader = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.workers,
        "pin_memory": True,
        "collate_fn": seg_dataset_collate,
        "worker_init_fn": partial(worker_init_fn, rank=0, seed=config.train.seed),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader
    )
    epoch_step = num_train // config.train.batch_size
    epoch_step_val = (num_val + config.train.batch_size - 1) // config.train.batch_size

    eval_callback = EvalCallback(
        model,
        config.data.input_shape,
        config.model.num_classes,
        val_lines,
        config.data.dataset_path,
        str(log_dir),
        cuda,
        eval_flag=config.train.eval_enabled,
        period=config.train.eval_period,
        dataset_type=config.data.dataset_type,
    )
    class_weights = np.ones(config.model.num_classes, dtype=np.float32)
    save_run_manifest(config, run_dir, split_dir, initial_lr, final_lr)

    show_config(
        model="DBSegNet",
        branch=config.model.branch,
        fusion_mode=config.model.fusion_mode,
        cross_attention=config.model.use_cross_attention,
        dataset_path=config.data.dataset_path,
        split_dir=str(split_dir),
        num_train=num_train,
        num_val=num_val,
        input_shape=config.data.input_shape,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        fp16=fp16,
        optimizer=config.train.optimizer,
        Init_lr=config.train.learning_rate,
        Init_lr_fit=initial_lr,
        Min_lr_fit=final_lr,
        weight_decay=config.train.weight_decay,
        augmentation=config.data.augmentation,
        no_aug_last_epochs=config.data.no_augmentation_last_epochs,
        eval_period=config.train.eval_period,
        seed=config.train.seed,
        save_dir=str(run_dir),
    )
    print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2))

    last_augmentation = None
    for epoch in range(config.train.epochs):
        no_aug_start = (
            config.train.epochs - config.data.no_augmentation_last_epochs
        )
        train_dataset.aug_mode = (
            config.data.no_augmentation_mode
            if epoch >= no_aug_start
            else config.data.augmentation
        )
        if train_dataset.aug_mode != last_augmentation:
            print(
                "Train augmentation mode -> %s (epoch %d/%d)"
                % (train_dataset.aug_mode, epoch + 1, config.train.epochs)
            )
            last_augmentation = train_dataset.aug_mode

        set_optimizer_lr(optimizer, lr_scheduler, epoch)
        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            train_loader,
            val_loader,
            config.train.epochs,
            cuda,
            config.train.dice_loss,
            config.train.focal_loss,
            class_weights,
            config.model.num_classes,
            fp16,
            scaler,
            config.train.save_period,
            str(run_dir),
            0,
            fusion_warmup_epoch=0,
            tr_aux_start_epoch=40,
            tr_aux_max_weight=0.0,
            use_boundary_loss=False,
            boundary_loss_weight=0.0,
            grad_clip_norm=0.0,
            dice_weight=config.train.dice_weight,
            label_smoothing=config.train.label_smoothing,
            model_ema=model_ema,
            ema_start_epoch=config.train.ema_start_epoch,
        )

    loss_history.writer.close()
