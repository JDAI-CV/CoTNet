import decimal
import numpy as np
from collections import deque

import torch
from config import cfg
from utils.timer import Timer
from utils.logger import logger_info
import utils.distributed as dist
from utils.distributed import sum_tensor

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 1.0 for k in topk]

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)

def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024

def float_to_decimal(data, prec=4):
    """Convert floats to decimals which allows for fixed width json."""
    if isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data

class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count

class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = (num_epochs - start_epoch) * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.solver.log_interval)
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0
        self.max_epoch = num_epochs
        self.start_epoch = start_epoch

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, loss, lr, mb_size):
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
    
    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = (cur_epoch - self.start_epoch) * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.max_epoch),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.solver.log_interval != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        info = "Epoch: {:s}, Iter: {:s}, loss: {:.4f}, lr: {:s}, time_avg: {:.4f}, eta: {:s}, mem: {:d}".format(\
            stats["epoch"], stats["iter"], stats["loss"], stats["lr"], stats["time_avg"], stats["eta"], stats["mem"])
        logger_info(info)

class TestMeter(object):
    def __init__(self):
        self.num_top1 = 0
        self.num_top5 = 0
        self.num_samples = 0

    def reset(self):
        self.num_top1 = 0
        self.num_top5 = 0
        self.num_samples = 0

    def update_stats(self, num_top1, num_top5, mb_size):
        self.num_top1 += num_top1
        self.num_top5 += num_top5
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch):
        if cfg.distributed:
            tensor_reduce = torch.tensor([self.num_top1 * 1.0, self.num_top5 * 1.0, self.num_samples * 1.0], device="cuda")
            tensor_reduce = sum_tensor(tensor_reduce)
            tensor_reduce = tensor_reduce.data.cpu().numpy()
            num_top1 = tensor_reduce[0]
            num_top5 = tensor_reduce[1]
            num_samples = tensor_reduce[2]
        else:
            num_top1 = self.num_top1
            num_top5 = self.num_top5
            num_samples = self.num_samples

        top1_acc = num_top1 * 1.0 / num_samples
        top5_acc = num_top5 * 1.0 / num_samples

        info = "Epoch: {:d}, top1_acc = {:.2%}, top5_acc = {:.2%} in {:d}".format(cur_epoch + 1, top1_acc, top5_acc, int(num_samples))
        logger_info(info)
        return top1_acc, top5_acc
