#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

import utils.utils


def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    # metrics = [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():   
        rs = model.propagate()
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))
            # pre_b, pre_i = model.evaluate(rs, user.todevice))
            pred_b -= 1e8 * train_mask_u_b.to(device)
            # pred_i = 1e8 * train_mask_u_i.to(device)
            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
                # metric(pred_i, ground_truth_u_i.to(device))
    print('Test: time={:d}s'.format(int(time() - start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    return metrics
