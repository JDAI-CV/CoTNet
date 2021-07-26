import os
from yacs.config import CfgNode as CN
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

_C = CN()
_C.root_dir = os.getcwd()                         # root dir

_C.seed = -1.0                                    # random seed (default: 42)
_C.logger_name = 'log'                            # log name
_C.amp = False                                    # use NVIDIA amp for mixed precision training
_C.num_gpus = 1
_C.distributed = False

# data
_C.data_loader = CN()
_C.data_loader.data_path = ''                     # path to dataset, data_dir
_C.data_loader.batch_size = 32                    # input batch size for training (default: 32)
_C.data_loader.vbatch_size = 32                   # validation batch size
_C.data_loader.workers = 0                        # how many training processes to use (default: 1)
_C.data_loader.pin_mem = False                    # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.data_loader.prefetcher = True                  # enable fast prefetcher
_C.data_loader.use_multi_epochs_loader = False    # use the multi-epochs-loader to save time at the beginning of every epoch
_C.data_loader.dataset = 'imagenet'               # imagenet, cifar10, cifar100

# model
_C.model = CN()
_C.model.name = 'resnet50'                        # Name of model to train
_C.model.pretrained = False                       # Start with pretrained version of specified network (if avail)
_C.model.initial_checkpoint = ''                  # Initialize model from this checkpoint (default: none)
_C.model.resume = ''                              # Resume full model and optimizer state from checkpoint (default: none)
_C.model.no_resume_opt = False                    # prevent resume of optimizer state when resuming model
_C.model.num_classes = 1000                       # number of label classes (default: 1000)
_C.model.gp = 'avg'                               # Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")
_C.model.drop = 0.0                               # Dropout rate (default: 0.)
_C.model.drop_path = 0.0                          # Drop path rate (default None)
_C.model.drop_block = 0.0                         # Drop block rate (default None)
_C.model.model_ema = False                        # Enable tracking moving average of model weights
_C.model.model_ema_force_cpu = False              # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.
_C.model.model_ema_decay = 0.9998                 # decay factor for model weights moving average (default: 0.9998)
_C.model.block_name = 'type1'

# BN
_C.BN = CN()
_C.BN.bn_tf = False                               # Use Tensorflow BatchNorm defaults for models that support it (default: False)
_C.BN.bn_momentum = -1.0                          # BatchNorm momentum override (if not None) default None
_C.BN.bn_eps = -1.0                               # BatchNorm epsilon override (if not None)  default None
_C.BN.sync_bn = False                             # Enable NVIDIA Apex or Torch synchronized BatchNorm.
_C.BN.dist_bn = ''                                # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
_C.BN.split_bn = False                            # Enable separate BN layers per augmentation split.


# augmentation
_C.augmentation = CN()
_C.augmentation.no_aug = False
_C.augmentation.scale = [0.08, 1.0]
_C.augmentation.ratio = [0.75, 1.333333333333]
_C.augmentation.hflip = 0.5
_C.augmentation.vflip = 0.0
_C.augmentation.interpolation = ''                # Image resize interpolation type (overrides model)
_C.augmentation.color_jitter = 0.4                # Color jitter factor (default: 0.4)
_C.augmentation.aa = ''                           # Use AutoAugment policy. "v0" or "original". (default None)
_C.augmentation.aug_splits = 0                    # Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.augmentation.reprob = 0.0                      # Random erase prob (default: 0.)
_C.augmentation.remode = 'const'                  # Random erase mode (default: "const")
_C.augmentation.recount = 1                       # Random erase count (default: 1)
_C.augmentation.resplit = False                   # Do not random erase first (clean) augmentation split
_C.augmentation.mixup = 0.0                       # mixup alpha, mixup enabled if > 0. (default: 0.)
_C.augmentation.mixup_off_epoch = 0               # turn off mixup after this epoch, disabled if 0 (default: 0)
_C.augmentation.cutmix = 0.0
_C.augmentation.cutmix_minmax = []
_C.augmentation.mixup_prob = 1.0
_C.augmentation.mixup_switch_prob = 0.5
_C.augmentation.mixup_mode = 'batch'

_C.augmentation.train_interpolation = 'random'    # Training interpolation (random, bilinear, bicubic default: "random")
_C.augmentation.tta = 0                           # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
_C.augmentation.img_size = -1                     # Image patch size (default: None => model default)
_C.augmentation.crop_pct = -1.0                   # Input image center crop percent (for validation only)
_C.augmentation.mean = []                         # Override mean pixel value of dataset
_C.augmentation.std = []                          # Override std deviation of of dataset



# loss
_C.loss = CN()
_C.loss.jsd = False                               # Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
_C.loss.smoothing = 0.1                           # label smoothing (default: 0.1)


# solver
_C.solver = CN()
_C.solver.opt = 'sgd'                             # Optimizer (default: "sgd")
_C.solver.opt_eps = 1e-8                          # Optimizer Epsilon (default: 1e-8)
_C.solver.momentum = 0.9                          # SGD momentum (default: 0.9)
_C.solver.weight_decay = 0.0001                   # weight decay (default: 0.0001)
_C.solver.sched = 'step'                          # LR scheduler (default: "step")
_C.solver.lr = 0.01                               # learning rate (default: 0.01)
_C.solver.lr_noise = []                           # learning rate noise on/off epoch percentages  default None
_C.solver.lr_noise_pct = 0.67                     # learning rate noise limit percent (default: 0.67)
_C.solver.lr_noise_std = 1.0                      # learning rate noise std-dev (default: 1.0)
_C.solver.lr_cycle_mul = 1.0                      # learning rate cycle len multiplier (default: 1.0)
_C.solver.lr_cycle_limit = 1                      # learning rate cycle limit
_C.solver.warmup_lr = 0.0001                      # warmup learning rate (default: 0.0001)
_C.solver.min_lr = 1e-5                           # lower lr bound for cyclic schedulers that hit 0 (1e-5)
_C.solver.epochs = 200                            # number of epochs to train (default: 2)
_C.solver.start_epoch = -1                        # manual epoch number (useful on restarts)  default None
_C.solver.decay_epochs = 30                       # epoch interval to decay LR
_C.solver.warmup_epochs = 3                       # epochs to warmup LR, if scheduler supports
_C.solver.cooldown_epochs = 10                    # epochs to cooldown LR at min_lr, after cyclic schedule ends
_C.solver.patience_epochs = 10                    # patience epochs for Plateau LR scheduler (default: 10)
_C.solver.decay_rate = 0.1                        # LR decay rate (default: 0.1)
_C.solver.log_interval = 50                       # how many batches to wait before logging training status
_C.solver.recovery_interval = 0                   # how many batches to wait before writing recovery checkpoint
_C.solver.clip_grad = -1.0
_C.solver.clip_mode = 'norm'

_C.solver.use_swa = False
_C.solver.swa_start = 75
_C.solver.swa_freq = 1

# eval
_C.eval = CN()
_C.eval.eval_metric = 'top1'                      # Best metric (default: "top1")


def pop_unused_value(cfg):
    if cfg.BN.bn_momentum < 0:
        cfg.BN.pop('bn_momentum')
    if cfg.BN.bn_eps < 0:
        cfg.BN.pop('bn_eps')
    if len(cfg.solver.lr_noise) == 0:
        cfg.solver.pop('lr_noise')
    if cfg.solver.start_epoch < 0:
        cfg.solver.pop('start_epoch')
    if cfg.model.drop_path == 0:
        cfg.model.pop('drop_path')
    if cfg.model.drop_block == 0:
        cfg.model.pop('drop_block')
    if len(cfg.augmentation.aa) == 0:
        cfg.augmentation.pop('aa')
    if cfg.augmentation.img_size <= 0:
        cfg.augmentation.pop('img_size')
    if cfg.augmentation.crop_pct <= 0:
        cfg.augmentation.pop('crop_pct')
    if len(cfg.augmentation.mean) == 0:
        cfg.augmentation.pop('mean')
    if len(cfg.augmentation.std) == 0:
        cfg.augmentation.pop('std')
    


def resolve_data_config(cfg, default_cfg={}, model=None):
    new_config = {}
    default_cfg = default_cfg
    if not default_cfg and model is not None and hasattr(model, 'default_cfg'):
        default_cfg = model.default_cfg

    # Resolve input/image size
    in_chans = 3
    input_size = (in_chans, 224, 224)
    if 'img_size' in cfg.augmentation and cfg.augmentation.img_size > 0:
        assert isinstance(cfg.augmentation.img_size, int)
        input_size = (in_chans, cfg.augmentation.img_size, cfg.augmentation.img_size)
    elif 'input_size' in default_cfg:
        input_size = default_cfg['input_size']
    new_config['input_size'] = input_size

    # resolve interpolation method
    new_config['interpolation'] = 'bicubic'
    if 'interpolation' in cfg.augmentation and len(cfg.augmentation.interpolation) > 0:
        new_config['interpolation'] = cfg.augmentation.interpolation
    elif 'interpolation' in default_cfg:
        new_config['interpolation'] = default_cfg['interpolation']

    # resolve dataset + model mean for normalization
    new_config['mean'] = IMAGENET_DEFAULT_MEAN
    if 'mean' in cfg.augmentation and len(cfg.augmentation.mean) > 0:
        mean = tuple(cfg.augmentation.mean)
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        new_config['mean'] = mean
    elif 'mean' in default_cfg:
        new_config['mean'] = default_cfg['mean']

    # resolve dataset + model std deviation for normalization
    new_config['std'] = IMAGENET_DEFAULT_STD
    if 'std' in cfg.augmentation and len(cfg.augmentation.std) > 0:
        std = tuple(cfg.augmentation.std)
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        new_config['std'] = std
    elif 'std' in default_cfg:
        new_config['std'] = default_cfg['std']

    # resolve default crop percentage
    new_config['crop_pct'] = DEFAULT_CROP_PCT
    if 'crop_pct' in cfg.augmentation and cfg.augmentation.crop_pct > 0:
        new_config['crop_pct'] = cfg.augmentation.crop_pct
    elif 'crop_pct' in default_cfg:
        new_config['crop_pct'] = default_cfg['crop_pct']

    return new_config