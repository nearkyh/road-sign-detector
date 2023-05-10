import os
import random
import numpy as np
import torch


def set_seed(seed, multi_gpu=False):
    os.environ['PYTHONHASHseed'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if multi_gpu:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(device)
        print(f'Using GPU: {device_name}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    return device
