""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler
from .plateau_lr import PlateauLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.solver.epochs

    if 'lr_noise' in cfg.solver and len(cfg.solver.lr_noise) > 0:
        lr_noise = cfg.solver.lr_noise
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if cfg.solver.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=cfg.solver.lr_cycle_mul,
            lr_min=cfg.solver.min_lr,
            decay_rate=cfg.solver.decay_rate,
            warmup_lr_init=cfg.solver.warmup_lr,
            warmup_t=cfg.solver.warmup_epochs,
            cycle_limit=cfg.solver.lr_cycle_limit,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=cfg.solver.lr_noise_pct,
            noise_std=cfg.solver.lr_noise_std,
            noise_seed=cfg.seed,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.solver.cooldown_epochs
    elif cfg.solver.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=cfg.solver.lr_cycle_mul,
            lr_min=cfg.solver.min_lr,
            warmup_lr_init=cfg.solver.warmup_lr,
            warmup_t=cfg.solver.warmup_epochs,
            cycle_limit=cfg.solver.lr_cycle_limit,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=cfg.solver.lr_noise_pct,
            noise_std=cfg.solver.lr_noise_std,
            noise_seed=cfg.seed,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.solver.cooldown_epochs
    elif cfg.solver.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=cfg.solver.decay_epochs,
            decay_rate=cfg.solver.decay_rate,
            warmup_lr_init=cfg.solver.warmup_lr,
            warmup_t=cfg.solver.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=cfg.solver.lr_noise_pct,
            noise_std=cfg.solver.lr_noise_std,
            noise_seed=cfg.seed,
        )
    elif cfg.solver.sched == 'plateau':
        mode = 'min' if 'loss' in cfg.eval.eval_metric else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=cfg.solver.decay_rate,
            patience_t=cfg.solver.patience_epochs,
            lr_min=cfg.solver.min_lr,
            mode=mode,
            warmup_lr_init=cfg.solver.warmup_lr,
            warmup_t=cfg.solver.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=cfg.solver.lr_noise_pct,
            noise_std=cfg.solver.lr_noise_std,
            noise_seed=cfg.seed,
        )

    return lr_scheduler, num_epochs
