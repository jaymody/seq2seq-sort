import os
import random
from functools import partialmethod

import numpy as np
import torch
from tqdm import tqdm


def score(true_expansion, pred_expansion):
    return int(true_expansion == pred_expansion)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def disable_tqdm():
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def enable_tqdm():
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
