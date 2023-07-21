import os
import random

import torch
import numpy as np
from sklearn.metrics import mean_squared_error


def MCRMSE(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)  # RMSE
        scores += [score]
    return np.mean(scores), scores


def get_score(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
    mcrmse, scores = MCRMSE(y_true, y_pred)
    return mcrmse, scores


def str2bool(v):
	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
