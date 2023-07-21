import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSE(nn.Module):
    def __init__(self, reduction="mean", eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()


def get_gamma(n_epochs):
    # http://www.yichang-cs.com/yahoo/cikm09_smoothdcg.pdf
    tau_1 = 0.999999
    return np.log(1/tau_1 - 1) / (1 - n_epochs/2)

def get_tau(epoch, n_epochs, gamma):
    # http://www.yichang-cs.com/yahoo/cikm09_smoothdcg.pdf
    return 1 / (1 + np.exp(gamma * (epoch - n_epochs / 2)))
    
class R2Loss(nn.Module):
    # https://aclanthology.org/2020.findings-emnlp.141.pdf
    def __init__(self, n_epochs, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.n_epochs = n_epochs
        self.gamma = get_gamma(n_epochs)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true, epoch):
        assert epoch > 0
        P_pred = F.softmax(y_pred, dim=0)
        P_true = F.softmax(y_true, dim=0)

        L_r = - torch.sum(P_true * torch.log(P_pred), dim=0)
        L_m = self.mse(y_pred, y_true)

        tau = get_tau(epoch, self.n_epochs, self.gamma)

        loss = tau*L_r + (1-tau)*L_m

        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()


class RDropMSE(nn.Module):
    def __init__(self, alpha, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred_1, y_pred_2, y_true):
        mse_r = self.mse(y_pred_1, y_pred_2)
        mse_1 = self.mse(y_pred_1, y_true)
        mse_2 = self.mse(y_pred_2, y_true)
        loss = mse_1 + mse_2 + self.alpha*mse_r
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
