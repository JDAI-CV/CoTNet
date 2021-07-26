import os
import sys
import pprint

import logging
from config import cfg
import utils.distributed as dist 

def setup_default_logging():
    logger = logging.getLogger(cfg.logger_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(cfg.root_dir, cfg.logger_name + '.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))

    return logger


def logger_info(data):
    if dist.is_master_proc():
        logger = logging.getLogger(cfg.logger_name)
        logger.info(data)

            