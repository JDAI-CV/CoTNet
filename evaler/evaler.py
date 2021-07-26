import os
from config import cfg
from datasets import Dataset, create_loader
from utils.meters import TestMeter, accuracy
import torch
import torchvision
import torchvision.transforms as transforms
from datasets.distributed_sampler import OrderedDistributedSampler


class Evaler(object):
    def __init__(self,data_config):
        super(Evaler, self).__init__()
        self.loader_eval = self.build_dataset(data_config)

    def build_dataset(self, data_config):
        eval_dir = os.path.join(cfg.data_loader.data_path, 'val')
        assert os.path.isdir(eval_dir)
        dataset_eval = Dataset(eval_dir)

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=cfg.data_loader.vbatch_size,
            is_training=False,
            use_prefetcher=cfg.data_loader.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=cfg.data_loader.workers,
            distributed=cfg.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=cfg.data_loader.pin_mem,
        )
        return loader_eval

    def __call__(self, epoch, model, amp_autocast):
        test_meter = TestMeter()
        model.eval()

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.loader_eval):
                if not cfg.data_loader.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
    
                with amp_autocast():
                    output = model(input)
    
                top1_num, top5_num = accuracy(output, target, topk=(1, 5))
                test_meter.update_stats(top1_num, top5_num, input.size(0))
                torch.cuda.synchronize()
                
            if cfg.distributed:
                torch.distributed.barrier()
    
            top1_acc, top5_acc = test_meter.log_iter_stats(epoch)
            return top1_acc, top5_acc