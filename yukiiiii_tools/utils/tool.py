import os
import sys
from typing import NoReturn

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .misc import mkdir_if_missing


class TensorboardLogger(object):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def __del__(self):
        self.writer.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def set_step(self, step: int = None) -> NoReturn:
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head: str = 'scalar', step: int = None, **kwargs) -> NoReturn:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step)

    def flush(self) -> NoReturn:
        self.writer.flush()


class ProgressMeter(object):
    # TODO: 测试show_func参数来代替写死的print是否可行
    def __init__(self, num_batches, meters, show_func=print, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.show_func = show_func

    def display(self, batch: str) -> NoReturn:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.show_func('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> NoReturn:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[float, int]) -> NoReturn:
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def write(self, msg) -> NoReturn:
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self) -> NoReturn:
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
