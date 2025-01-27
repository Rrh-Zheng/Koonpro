import torch
import numpy as np
import torch.nn.functional as F
import random
import os
import json
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def varying_multiply(y, the, delta_t):
    phi =1
    cos_the = torch.multiply(phi, torch.cos(the * delta_t))
    sin_the = torch.multiply(phi, torch.sin(the * delta_t))
    vec1 = torch.stack((cos_the, -sin_the), dim=2).view(the.shape[0], -1)
    vec2 = torch.stack((sin_the, cos_the), dim=2).view(the.shape[0], -1)
    r1 = torch.multiply(y, vec1)
    r2 = torch.multiply(y, vec2)
    r2 = r2.view(the.shape[0], the.shape[1], 2, 1)
    r2 = torch.sum(r2, dim=2)
    r1 = r1.view(the.shape[0], the.shape[1], 2, 1)
    r1 = torch.sum(r1, dim=2)
    r = torch.stack((r1, r2), dim=2).view(the.shape[0], -1)
    return r

class MLN_MLP(torch.nn.Module):
    def __init__(self, layer_width):
        super(MLN_MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        for j in range(len(layer_width) - 1):
            self.layers.append(torch.nn.Linear(layer_width[j], layer_width[j + 1]))
            # if j < len(layer_width) - 2:
            #     self.layers.append(torch.nn.ReLU(inplace=False))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, q_context, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, q_context, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint( val_loss, model, q_context, path)
            self.counter = 0
        return self.counter

    def save_checkpoint(self, val_loss, model, q_context, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model_path = path + '/' + 'checkpoint.pth'
        torch.save(model.state_dict(), model_path)
        q_context_path = path + '/q_context.npz'
        np.savez(q_context_path, q_context=q_context)
        self.val_loss_min = val_loss

def get_back(x_list_mu, x_list_sigma, y):
    x_list_mu = x_list_mu[:, :, :, -1]
    x_list_sigma = x_list_sigma[:, :, :, -1]
    y = y[:, :, :, -1]
    return x_list_mu, x_list_sigma, y
