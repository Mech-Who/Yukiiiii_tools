import os
import errno
import random
from typing import NoReturn

import numpy as np
import torch


def setup_system(seed: int, cudnn_benchmark: bool = True, cudnn_deterministic: bool = True) -> NoReturn:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def mkdir_if_missing(dir_path: str) -> NoReturn:
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
