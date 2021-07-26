import argparse
import time
import yaml
import os
import sys
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from config import cfg, resolve_data_config, pop_unused_value
from datasets import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from models import create_model, resume_checkpoint, convert_splitbn_model, load_checkpoint, model_parameters
from utils import *
from loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from optim import create_optimizer
from scheduler import create_scheduler
from utils import ApexScaler, NativeScaler
from utils.logger import logger_info, setup_default_logging
import utils.distributed as dist
from utils.flops_counter import get_model_complexity_info
from evaler.evaler import Evaler
#from torchcontrib.optim import SWA

if cfg.amp == True:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

torch.backends.cudnn.benchmark = True

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Imagenet Model')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def setup_model():
    model = create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=cfg.model.drop,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=cfg.model.drop_path if 'drop_path' in cfg.model else None,
        drop_block_rate=cfg.model.drop_block if 'drop_block' in cfg.model else None,
        global_pool=cfg.model.gp,
        bn_tf=cfg.BN.bn_tf,
        bn_momentum=cfg.BN.bn_momentum if 'bn_momentum' in cfg.BN else None,
        bn_eps=cfg.BN.bn_eps if 'bn_eps' in cfg.BN else None,
        checkpoint_path=cfg.model.initial_checkpoint)
    data_config = resolve_data_config(cfg, model=model)

    flops_count, params_count = get_model_complexity_info(model, data_config['input_size'], as_strings=True,
        print_per_layer_stat=False, verbose=False)
    logger_info('Model %s created, flops_count: %s, param count: %s' % (cfg.model.name, flops_count, params_count))

    if cfg.BN.split_bn:
        assert cfg.augmentation.aug_splits > 1 or cfg.augmentation.resplit
        model = convert_splitbn_model(model, max(cfg.augmentation.aug_splits, 2))
    model.cuda()
    return model, data_config

def setup_resume(local_rank, model, optimizer):
    loss_scaler = None
    if cfg.amp == True:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
    else:
        logger_info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if cfg.model.resume:
        resume_epoch = resume_checkpoint(
            model, cfg.model.resume,
            optimizer=None if cfg.model.no_resume_opt else optimizer,
            loss_scaler=None if cfg.model.no_resume_opt else loss_scaler,
            log_info=local_rank == 0)

    if cfg.distributed:
        if cfg.BN.sync_bn:
            assert not cfg.BN.split_bn
            try:
                if cfg.amp:
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger_info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logger_info('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if cfg.amp:
            # Apex DDP preferred unless native amp is activated
            logger_info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            logger_info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    model_ema = None
    if cfg.model.model_ema == True:
        model_ema = ModelEmaV2(
            unwrap_model(model), 
            decay=cfg.model.model_ema_decay,
            device='cpu' if cfg.model.model_ema_force_cpu else None
        )
        if cfg.model.resume:
            load_checkpoint(model_ema.module, cfg.model.resume, use_ema=True)

    return model, model_ema, optimizer, resume_epoch, loss_scaler

def setup_scheduler(optimizer, resume_epoch):
    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)
    start_epoch = 0
    if 'start_epoch' in cfg.solver:
        # a specified start_epoch will always override the resume epoch
        start_epoch = cfg.solver.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    return lr_scheduler, start_epoch, num_epochs

def setup_loader(data_config):
    train_dir = os.path.join(cfg.data_loader.data_path, 'train')
    assert os.path.exists(train_dir)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = cfg.augmentation.mixup > 0 or cfg.augmentation.cutmix > 0. or len(cfg.augmentation.cutmix_minmax) > 0
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=cfg.augmentation.mixup, cutmix_alpha=cfg.augmentation.cutmix, cutmix_minmax=cfg.augmentation.cutmix_minmax,
            prob=cfg.augmentation.mixup_prob, switch_prob=cfg.augmentation.mixup_switch_prob, mode=cfg.augmentation.mixup_mode,
            label_smoothing=cfg.loss.smoothing, num_classes=cfg.model.num_classes)
        if cfg.data_loader.prefetcher:
            assert not cfg.augmentation.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if cfg.augmentation.aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=cfg.augmentation.aug_splits)

    train_interpolation = cfg.augmentation.train_interpolation
    if cfg.augmentation.no_aug:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=cfg.data_loader.batch_size,
        is_training=True,
        use_prefetcher=cfg.data_loader.prefetcher,
        no_aug=cfg.augmentation.no_aug,
        re_prob=cfg.augmentation.reprob,
        re_mode=cfg.augmentation.remode,
        re_count=cfg.augmentation.recount,
        re_split=cfg.augmentation.resplit,
        scale=cfg.augmentation.scale,
        ratio=cfg.augmentation.ratio,
        hflip=cfg.augmentation.hflip,
        vflip=cfg.augmentation.vflip,
        color_jitter=cfg.augmentation.color_jitter if cfg.augmentation.color_jitter > 0 else None,
        auto_augment=cfg.augmentation.aa if 'aa' in cfg.augmentation else None,
        num_aug_splits=cfg.augmentation.aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.data_loader.workers,
        distributed=cfg.distributed,
        collate_fn=collate_fn,
        pin_memory=cfg.data_loader.pin_mem,
        use_multi_epochs_loader=cfg.data_loader.use_multi_epochs_loader
    )
  
    return loader_train, mixup_active, mixup_fn

def setup_loss(mixup_active):
    if cfg.loss.jsd:
        assert cfg.augmentation.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=cfg.augmentation.aug_splits, smoothing=cfg.loss.smoothing).cuda()
    elif mixup_active: # smoothing is handled with mixup label transform 
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif cfg.loss.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=cfg.loss.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()

    return train_loss_fn

def setup_env(args):
    if args.folder is not None:
        cfg.merge_from_file(os.path.join(args.folder, 'config.yaml'))
    cfg.root_dir = args.folder

    setup_default_logging()

    world_size = 1
    rank = 0  # global rank
    cfg.distributed = torch.cuda.device_count() > 1

    if cfg.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    cfg.num_gpus = world_size

    pop_unused_value(cfg)
    cfg.freeze()

    if cfg.distributed:
        logger_info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (rank, cfg.num_gpus))
    else:
        logger_info('Training with a single process on %d GPUs.' % cfg.num_gpus)
    torch.manual_seed(cfg.seed + rank)

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, cfg,
        lr_scheduler=None, saver=None, train_meter=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if cfg.augmentation.mixup_off_epoch and epoch >= cfg.augmentation.mixup_off_epoch:
        if cfg.data_loader.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    model.train()
    num_updates = epoch * len(loader)
    train_meter.iter_tic()

    for batch_idx, (input, target) in enumerate(loader):
        if not cfg.data_loader.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if cfg.solver.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in cfg.solver.clip_mode),
                    value=cfg.solver.clip_grad, mode=cfg.solver.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        #if (cfg.solver.use_swa) and (cfg.solver.swa_start >= epoch) and (epoch % cfg.solver.swa_freq == 0):
        #    optimizer.update_swa()

        torch.cuda.synchronize()
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=None)
        
        num_updates += 1
        loss = dist.scaled_all_reduce([loss.data])[0]
        mb_size = input.size(0) * cfg.num_gpus
        lr_str = str(list(set([param_group['lr'] for param_group in optimizer.param_groups])))
        train_meter.update_stats(loss.item(), lr_str, mb_size)
        train_meter.iter_toc()
        train_meter.log_iter_stats(epoch, batch_idx)
        train_meter.iter_tic()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    train_meter.reset()

def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    setup_env(args)
    if cfg.distributed:
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    model, data_config = setup_model()
    optimizer = create_optimizer(cfg, model)

    amp_autocast = suppress  # do nothing
    
    model, model_ema, optimizer, resume_epoch, loss_scaler = setup_resume(args.local_rank, model, optimizer)
    lr_scheduler, start_epoch, num_epochs = setup_scheduler(optimizer, resume_epoch)

    #if cfg.solver.use_swa == True:
    #    optimizer = SWA(optimizer)
    
    loader_train, mixup_active, mixup_fn = setup_loader(data_config)
    train_loss_fn = setup_loss(mixup_active)
    train_meter = TrainMeter(start_epoch, num_epochs, len(loader_train))
    evaler = Evaler(data_config)

    best_metric = None
    best_epoch = None
    saver = None
    if global_rank == 0:
        snapshot_dir = os.path.join(cfg.root_dir, 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, cfg=cfg, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=snapshot_dir, recovery_dir=snapshot_dir, decreasing=False)

    try:
        for epoch in range(start_epoch, num_epochs):
            if cfg.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, cfg,
                lr_scheduler=lr_scheduler, saver=saver, train_meter=train_meter,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if cfg.distributed and cfg.BN.dist_bn in ('broadcast', 'reduce'):
                logger_info("Distributing BatchNorm running means and vars")
                distribute_bn(model, cfg.num_gpus, cfg.BN.dist_bn == 'reduce')

            top1_acc, top5_acc = evaler(epoch, model, amp_autocast=amp_autocast)

            if model_ema is not None and not cfg.model.model_ema_force_cpu:
                if cfg.distributed and cfg.BN.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, cfg.num_gpus, cfg.BN.dist_bn == 'reduce')
                top1_acc, top5_acc = evaler(epoch, model_ema.module, amp_autocast=amp_autocast)

            if (saver is not None) and (epoch + 1) % cfg.solver.recovery_interval == 0:
                saver.save_recovery(epoch + 1)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, None)

            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch + 1, metric=top1_acc)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logger_info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    #if cfg.solver.use_swa:
    #    optimizer.swap_swa_sgd()
    #    top1_acc, top5_acc = evaler(epoch, model, amp_autocast=amp_autocast)
    #    saver.save_recovery(epoch + 2)


if __name__ == '__main__':
    main()
