""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
from .adamw import AdamW

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(cfg, model, filter_bias_and_bn=True):
    if cfg.amp == True:
        from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    opt_lower = cfg.solver.opt.lower()
    weight_decay = cfg.solver.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert cfg.amp and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(
            parameters, lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters, lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = AdamW(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'nadam':
        optimizer = Nadam(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'radam':
        optimizer = RAdam(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps,
            delta=0.1, wd_ratio=0.01, nesterov=True)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(
            parameters, lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=weight_decay, 
            eps=cfg.solver.opt_eps, nesterov=True)        
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, lr=cfg.solver.lr, alpha=0.9, eps=cfg.solver.opt_eps,
            momentum=cfg.solver.momentum, weight_decay=weight_decay)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, lr=cfg.solver.lr, alpha=0.9, eps=cfg.solver.opt_eps,
            momentum=cfg.solver.momentum, weight_decay=weight_decay)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'fusedsgd':
        optimizer = FusedSGD(
            parameters, lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'fusedmomentum':
        optimizer = FusedSGD(
            parameters, lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(
            parameters, lr=cfg.solver.lr, adam_w_mode=False, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(
            parameters, lr=cfg.solver.lr, adam_w_mode=True, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, lr=cfg.solver.lr, weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    elif opt_lower == 'fusednovograd':
        optimizer = FusedNovoGrad(
            parameters, lr=cfg.solver.lr, betas=(0.95, 0.98), weight_decay=weight_decay, eps=cfg.solver.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer