import os, random
import time
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD, Adam, AdamW
# from tensorboardX import SummaryWriter
from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology as ski_morph
from skimage import measure
from skimage.measure import label
from scipy.optimize import linear_sum_assignment
from numba import jit
from typing import List

def save_checkpoint(save_path, model, epoch):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({'epoch': epoch + 1, 'state_dict': state_dict}, save_path)

def load_checkpoint(model, model_path):
    if not os.path.isfile(model_path):
        raise ValueError('Invalid checkpoint file: {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    try:
        epoch = checkpoint['epoch']
        tmp_state_dict = checkpoint['state_dict']
        print('loaded {}, epoch {}'.format(model_path, epoch))
    except:
        # The most naive way for serialization (especially for efficientdet)
        tmp_state_dict = checkpoint

    # create state_dict
    state_dict = {}

    # convert data_parallal to model
    for k in tmp_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = tmp_state_dict[k]
        else:
            state_dict[k] = tmp_state_dict[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                try:
                    tmp = torch.zeros(model_state_dict[k].shape)  # create tensor with zero filled
                    tmp[:state_dict[k].shape[0]] = state_dict[k]  # fill valid
                    state_dict[k] = tmp
                    print('Load parameter partially {}, required shape {}, loaded shape {}'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                except:
                    print('Remain parameter (as random) {}'.format(k))  # when loaded state_dict has larger tensor
                    state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            # print('No param {}'.format(k))
            state_dict[k] = model_state_dict[k]

    # load state_dict
    model.load_state_dict(state_dict, strict=False)

    return model