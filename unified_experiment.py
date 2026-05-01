"""
Unified Multi-Resolution Experiment for Interpretable Varroa Mite Detection
============================================================================
Trains 21 CNN models (5 architectures × 4 feature-map resolutions + RegNet
baseline) across 12 preprocessing variants, with full Grad-CAM++ XAI analysis.

Three phases:
  Phase 1  K-fold cross-validation (3-fold stratified)
  Phase 2  Single-split (70/20/10) training + Grad-CAM++ heatmaps + inference latency
  Phase 3  XAI metrics (annotation-free & localization)

Architectures:
  ShuffleNet-V2 (x1.0, x0.5), MobileNetV3-Small, EfficientNet-B0, VarroaNet
  + RegNet-Y-400MF (7×7 only, excluded from multi-resolution analysis)

Feature-map resolutions: 7×7 (base), 14×14, 28×28, 56×56
  Higher resolutions achieved by replacing stride-2 → stride-1 in downsampling blocks.

Per-model logs:  outputs/logs/{model}/epoch_log.csv, kfold_summary.csv, ...
Merged outputs:  outputs/logs/merged_*.csv
Weights:         outputs/weights/d*/best_{model}.pth, last_{model}.pth
                 outputs/weights_kfold/d*/best_{model}_fold{n}.pth
Grad-CAM++:      outputs/gradcam/d*/{model}/*.npy

Usage:
  python unified_experiment.py --gpu 0
  python unified_experiment.py --gpu 1 --phase kfold gradcam --resume
  python unified_experiment.py --gpu 0 --models varroanet_56 --phase gradcam
  python unified_experiment.py --resolution 56x56
  python unified_experiment.py --phase xai
  python unified_experiment.py --list-models

Author:  Hong-Gu Lee (hgl@kangwon.ac.kr)
License: MIT
"""
import os
import sys
import random
import time
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix as sklearn_cm)
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    # 7×7 base
    'shufflenet_v2_x1_0':     {'res': '7x7',   'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'shufflenet_v2_x0_5':     {'res': '7x7',   'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'mobilenet_v3_small':     {'res': '7x7',   'batch': 32, 'attn': 'SE_builtin',     'arch': 'mobilenet'},
    'efficientnet_b0':        {'res': '7x7',   'batch': 32, 'attn': 'SE_builtin',     'arch': 'efficientnet'},
    'regnet_y_400mf':         {'res': '7x7',   'batch': 32, 'attn': 'none',           'arch': 'regnet'},
    'varroanet':              {'res': '7x7',   'batch': 32, 'attn': 'custom_channel', 'arch': 'varroanet'},
    # 14×14
    'shufflenet_v2_x1_0_hr':  {'res': '14x14', 'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'shufflenet_v2_x0_5_hr':  {'res': '14x14', 'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'mobilenet_v3_small_hr':  {'res': '14x14', 'batch': 32, 'attn': 'SE_builtin',     'arch': 'mobilenet'},
    'efficientnet_b0_hr':     {'res': '14x14', 'batch': 32, 'attn': 'SE_builtin',     'arch': 'efficientnet'},
    'varroanet_hr':           {'res': '14x14', 'batch': 32, 'attn': 'custom_channel', 'arch': 'varroanet'},
    # 28×28
    'shufflenet_v2_x1_0_28':  {'res': '28x28', 'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'shufflenet_v2_x0_5_28':  {'res': '28x28', 'batch': 32, 'attn': 'none',           'arch': 'shufflenet'},
    'mobilenet_v3_small_28':  {'res': '28x28', 'batch': 32, 'attn': 'SE_builtin',     'arch': 'mobilenet'},
    'efficientnet_b0_28':     {'res': '28x28', 'batch': 32, 'attn': 'SE_builtin',     'arch': 'efficientnet'},
    'varroanet_28':           {'res': '28x28', 'batch': 32, 'attn': 'custom_channel', 'arch': 'varroanet'},
    # 56×56
    'shufflenet_v2_x1_0_56':  {'res': '56x56', 'batch': 8,  'attn': 'none',           'arch': 'shufflenet'},
    'shufflenet_v2_x0_5_56':  {'res': '56x56', 'batch': 8,  'attn': 'none',           'arch': 'shufflenet'},
    'mobilenet_v3_small_56':  {'res': '56x56', 'batch': 8,  'attn': 'SE_builtin',     'arch': 'mobilenet'},
    'efficientnet_b0_56':     {'res': '56x56', 'batch': 8,  'attn': 'SE_builtin',     'arch': 'efficientnet'},
    'varroanet_56':           {'res': '56x56', 'batch': 8,  'attn': 'custom_channel', 'arch': 'varroanet'},
}

DATASETS = [
    'dataset_1', 'dataset_1_normalized', 'dataset_1_resized',
    'dataset_1_resized_set', 'dataset_1_normalized_resized',
    'dataset_1_normalized_resized_set',
    'dataset_2_deblurred', 'dataset_2_deblurred_normalized',
    'dataset_2_deblurred_resized', 'dataset_2_deblurred_resized_set',
    'dataset_2_deblurred_normalized_resized',
    'dataset_2_deblurred_normalized_resized_set',
]

DS_MAP = {
    'd1': 'dataset_1', 'd2': 'dataset_1_normalized',
    'd3': 'dataset_1_resized', 'd4': 'dataset_1_resized_set',
    'd5': 'dataset_1_normalized_resized', 'd6': 'dataset_1_normalized_resized_set',
    'd7': 'dataset_2_deblurred', 'd8': 'dataset_2_deblurred_normalized',
    'd9': 'dataset_2_deblurred_resized', 'd10': 'dataset_2_deblurred_resized_set',
    'd11': 'dataset_2_deblurred_normalized_resized',
    'd12': 'dataset_2_deblurred_normalized_resized_set',
}
DS_MAP_REV = {v: k for k, v in DS_MAP.items()}

# Hyperparameters
N_FOLDS = 3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 500
PATIENCE = 12
MIN_DELTA = 0.0005
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
DATA_ROOT = './dataset'
LOG_DIR = 'outputs/logs'


def model_log_dir(model_name):
    """Per-model log directory: outputs/logs/{model_name}/"""
    return os.path.join(LOG_DIR, model_name)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# VarroaNet: ShuffleNet-V2 + Channel Attention
# ═══════════════════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention (Hu et al., 2018)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class VarroaNet(nn.Module):
    """ShuffleNet-V2-x1.0 + channel attention after each stage. Target layer: conv5."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.shufflenet_v2_x1_0(weights=weights)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5
        self.attn2 = ChannelAttention(116)
        self.attn3 = ChannelAttention(232)
        self.attn4 = ChannelAttention(464)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.attn2(self.stage2(x))
        x = self.attn3(self.stage3(x))
        x = self.attn4(self.stage4(x))
        x = self.conv5(x)
        x = x.mean([2, 3])
        return self.fc(x)


class VarroaNetHR(VarroaNet):
    """VarroaNet with 14×14 feature maps (stage4 stride removed)."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__(num_classes, pretrained=False)
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.shufflenet_v2_x1_0(weights=weights)
        _modify_shufflenet_stride(base.stage4[0], pretrained)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5


class VarroaNet28(VarroaNet):
    """VarroaNet with 28×28 feature maps (stage3 + stage4 strides removed)."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__(num_classes, pretrained=False)
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.shufflenet_v2_x1_0(weights=weights)
        _modify_shufflenet_stride(base.stage3[0], pretrained)
        _modify_shufflenet_stride(base.stage4[0], pretrained)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5


class VarroaNet56(VarroaNet):
    """VarroaNet with 56×56 feature maps (stage2 + stage3 + stage4 strides removed)."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__(num_classes, pretrained=False)
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = models.shufflenet_v2_x1_0(weights=weights)
        _modify_shufflenet_stride(base.stage2[0], pretrained)
        _modify_shufflenet_stride(base.stage3[0], pretrained)
        _modify_shufflenet_stride(base.stage4[0], pretrained)
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5


# ═══════════════════════════════════════════════════════════════════════════════
# Stride Modification Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _modify_shufflenet_stride(block, copy_weights=True):
    """Replace stride-2 with stride-1 in ShuffleNet InvertedResidual block."""
    old1 = block.branch1[0]
    new1 = nn.Conv2d(old1.in_channels, old1.out_channels,
                     kernel_size=old1.kernel_size, stride=(1, 1),
                     padding=old1.padding, groups=old1.groups, bias=False)
    if copy_weights:
        new1.weight = old1.weight
    block.branch1[0] = new1

    old2 = block.branch2[3]
    new2 = nn.Conv2d(old2.in_channels, old2.out_channels,
                     kernel_size=old2.kernel_size, stride=(1, 1),
                     padding=old2.padding, groups=old2.groups, bias=False)
    if copy_weights:
        new2.weight = old2.weight
    block.branch2[3] = new2


def _modify_conv_stride(conv, copy_weights=True):
    """Replace stride-2 with stride-1 in a depthwise convolution layer."""
    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=(1, 1),
        padding=conv.padding, groups=conv.groups,
        bias=conv.bias is not None
    )
    if copy_weights:
        new_conv.weight = conv.weight
        if conv.bias is not None:
            new_conv.bias = conv.bias
    return new_conv


# ═══════════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════════
def _make_shufflenet(version, resolution, num_classes=2, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = (models.shufflenet_v2_x1_0(weights=weights) if version == 'x1_0'
             else models.shufflenet_v2_x0_5(weights=weights))
    if resolution == '14x14':
        _modify_shufflenet_stride(model.stage4[0], pretrained)
    elif resolution == '28x28':
        _modify_shufflenet_stride(model.stage3[0], pretrained)
        _modify_shufflenet_stride(model.stage4[0], pretrained)
    elif resolution == '56x56':
        _modify_shufflenet_stride(model.stage2[0], pretrained)
        _modify_shufflenet_stride(model.stage3[0], pretrained)
        _modify_shufflenet_stride(model.stage4[0], pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _make_mobilenetv3s(resolution, num_classes=2, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    # stride mods accumulate from deepest to shallowest
    if resolution in ('14x14', '28x28', '56x56'):
        old9 = model.features[9].block[1][0]
        model.features[9].block[1][0] = _modify_conv_stride(old9, pretrained)
    if resolution in ('28x28', '56x56'):
        old4 = model.features[4].block[1][0]
        model.features[4].block[1][0] = _modify_conv_stride(old4, pretrained)
    if resolution == '56x56':
        old2 = model.features[2].block[1][0]
        model.features[2].block[1][0] = _modify_conv_stride(old2, pretrained)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def _make_efficientnetb0(resolution, num_classes=2, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if resolution in ('14x14', '28x28', '56x56'):
        old6 = model.features[6][0].block[1][0]
        model.features[6][0].block[1][0] = _modify_conv_stride(old6, pretrained)
    if resolution in ('28x28', '56x56'):
        old4 = model.features[4][0].block[1][0]
        model.features[4][0].block[1][0] = _modify_conv_stride(old4, pretrained)
    if resolution == '56x56':
        old3 = model.features[3][0].block[1][0]
        model.features[3][0].block[1][0] = _modify_conv_stride(old3, pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_model(model_name, num_classes=2, pretrained=True):
    """Unified model dispatcher for all 21 models."""
    cfg = MODEL_CONFIG[model_name]
    res = cfg['res']

    # ShuffleNet variants
    if model_name.startswith('shufflenet_v2_x1_0'):
        return _make_shufflenet('x1_0', res, num_classes, pretrained)
    if model_name.startswith('shufflenet_v2_x0_5'):
        return _make_shufflenet('x0_5', res, num_classes, pretrained)

    # MobileNetV3-Small
    if model_name.startswith('mobilenet_v3_small'):
        return _make_mobilenetv3s(res, num_classes, pretrained)

    # EfficientNet-B0
    if model_name.startswith('efficientnet_b0'):
        return _make_efficientnetb0(res, num_classes, pretrained)

    # RegNet-Y-400MF (7×7 only)
    if model_name == 'regnet_y_400mf':
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.regnet_y_400mf(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # VarroaNet variants
    if model_name == 'varroanet':
        return VarroaNet(num_classes, pretrained)
    if model_name == 'varroanet_hr':
        return VarroaNetHR(num_classes, pretrained)
    if model_name == 'varroanet_28':
        return VarroaNet28(num_classes, pretrained)
    if model_name == 'varroanet_56':
        return VarroaNet56(num_classes, pretrained)

    raise ValueError(f'Unknown model: {model_name}')


def get_target_layer(model, model_name):
    """Grad-CAM++ target layer for each architecture."""
    cfg = MODEL_CONFIG[model_name]
    arch = cfg['arch']
    if arch in ('shufflenet', 'varroanet'):
        return model.conv5
    elif arch == 'mobilenet':
        return model.features[-1]
    elif arch == 'efficientnet':
        return model.features[-1]
    elif arch == 'regnet':
        return model.trunk_output.block4
    raise ValueError(f'No target layer for {model_name}')


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Logging & Resume
# ═══════════════════════════════════════════════════════════════════════════════
EPOCH_FIELDS = [
    'model', 'dataset', 'phase', 'fold', 'epoch',
    'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'lr', 'best_val_loss', 'patience_counter', 'is_best', 'timestamp',
]

KFOLD_SUMMARY_FIELDS = [
    'model', 'dataset', 'fold', 'best_epoch', 'total_epochs', 'final_lr',
    'best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc',
    'final_train_loss', 'final_train_acc', 'final_val_loss', 'final_val_acc',
    'test_acc', 'test_f1', 'test_prec', 'test_rec',
    'tp', 'fp', 'tn', 'fn',
    'train_size', 'test_size', 'time_sec',
]

SINGLESPLIT_SUMMARY_FIELDS = [
    'model', 'dataset', 'best_epoch', 'total_epochs', 'final_lr',
    'best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc',
    'final_train_loss', 'final_train_acc', 'final_val_loss', 'final_val_acc',
    'test_acc', 'test_f1', 'test_prec', 'test_rec',
    'tp', 'fp', 'tn', 'fn',
    'infer_mean_ms', 'infer_median_ms', 'infer_std_ms',
    'infer_p1_ms', 'infer_p5_ms', 'infer_p95_ms', 'infer_p99_ms',
    'infer_total_ms', 'infer_n_samples',
    'train_size', 'val_size', 'test_size', 'n_gradcam', 'time_sec',
]


class CSVLogger:
    """Append-mode CSV writer with lazy header."""
    def __init__(self, path, fieldnames):
        self.path = os.path.normpath(path)
        self.fieldnames = fieldnames
        self._ensure_dir()
        self._header_written = os.path.exists(self.path) and os.path.getsize(self.path) > 0

    def _ensure_dir(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, row_dict):
        self._ensure_dir()
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row_dict)

    def flush_rows(self, rows):
        self._ensure_dir()
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)


def load_completed(csv_path, key_fields):
    """Load set of completed (key_field_values) tuples for resume."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = tuple(row[k] for k in key_fields)
                completed.add(key)
    return completed


def load_completed_epochs(csv_path):
    """Load set of (dataset, phase, fold, epoch) already logged for a model's epoch_log."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row['dataset'], row['phase'], row['fold'], row['epoch'])
                completed.add(key)
    return completed


# ═══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


def evaluate_full(model, loader, criterion, device):
    """Evaluate with full metrics + per-sample predictions for confusion matrix."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    prec = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    cm = sklearn_cm(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'loss': loss_sum / total, 'acc': acc,
        'f1': f1, 'prec': prec, 'rec': rec,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.SiLU)):
            module.inplace = False
    return model


def measure_inference_latency(model, loader, device):
    """Per-sample inference latency (ms). Returns P1-P99 stats, excluding outliers."""
    model.eval()
    latencies = []
    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            for i in range(inputs.size(0)):
                single = inputs[i:i+1]
                if use_cuda:
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = model(single)
                if use_cuda:
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(latencies)
    # Trim P1-P99 for stable stats
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    trimmed = arr[(arr >= p1) & (arr <= p99)]

    return {
        'infer_mean_ms': round(float(np.mean(trimmed)), 3),
        'infer_median_ms': round(float(np.median(trimmed)), 3),
        'infer_std_ms': round(float(np.std(trimmed)), 3),
        'infer_p1_ms': round(float(p1), 3),
        'infer_p5_ms': round(float(np.percentile(arr, 5)), 3),
        'infer_p95_ms': round(float(np.percentile(arr, 95)), 3),
        'infer_p99_ms': round(float(p99), 3),
        'infer_total_ms': round(float(np.sum(arr)), 3),
        'infer_n_samples': len(arr),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop (shared by K-fold and single-split)
# ═══════════════════════════════════════════════════════════════════════════════
def train_loop(model, train_loader, val_loader, device, epoch_logger,
               model_name, dataset_name, phase, fold, existing_epochs=None):
    """
    Train with AdamW + ReduceLROnPlateau + early stopping.
    Logs every epoch to epoch_logger (per-model file).
    existing_epochs: set of (dataset, phase, fold, epoch) already logged — skip these on resume.
    Returns dict with best_state, last_state, and all tracked metrics.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    epoch_rows = []

    # Track metrics at best epoch and final epoch
    best_train_loss, best_train_acc = 0.0, 0.0
    best_val_acc = 0.0
    train_loss, train_acc = 0.0, 0.0
    val_loss, val_acc = 0.0, 0.0
    current_lr = LEARNING_RATE

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_full(model, val_loader, criterion, device)
        val_loss, val_acc = val_metrics['loss'], val_metrics['acc']

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss - MIN_DELTA
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_rows.append({
            'model': model_name, 'dataset': dataset_name,
            'phase': phase, 'fold': fold, 'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 2),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 2),
            'lr': current_lr,
            'best_val_loss': round(best_val_loss, 6),
            'patience_counter': patience_counter,
            'is_best': int(is_best),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        })

        if patience_counter >= PATIENCE:
            break

    # Filter out already-logged epochs on resume, then flush
    if existing_epochs:
        epoch_rows = [r for r in epoch_rows
                      if (r['dataset'], r['phase'], str(r['fold']), str(r['epoch']))
                      not in existing_epochs]
    epoch_logger.flush_rows(epoch_rows)
    total_epochs = epoch + 1

    # Save last state before loading best
    last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {
        'best_state': best_state,
        'last_state': last_state,
        'best_epoch': best_epoch,
        'total_epochs': total_epochs,
        'final_lr': current_lr,
        'best_train_loss': round(best_train_loss, 6),
        'best_train_acc': round(best_train_acc, 2),
        'best_val_loss': round(best_val_loss, 6),
        'best_val_acc': round(best_val_acc, 2),
        'final_train_loss': round(train_loss, 6),
        'final_train_acc': round(train_acc, 2),
        'final_val_loss': round(val_loss, 6),
        'final_val_acc': round(val_acc, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: K-fold Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_kfold(model_list, device, resume=False):
    print('\n' + '=' * 70)
    print('PHASE 1: K-fold Cross-Validation')
    total = len(model_list) * len(DATASETS) * N_FOLDS
    print(f'  Models: {len(model_list)} | Datasets: {len(DATASETS)} | Folds: {N_FOLDS}')
    print(f'  Total training runs: {total}')
    print('=' * 70)

    done_count = 0
    for dataset_name in DATASETS:
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = ImageFolder(
            root=os.path.join(DATA_ROOT, dataset_name), transform=transform)
        labels = [s[1] for s in full_dataset.samples]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        folds = list(skf.split(range(len(full_dataset)), labels))

        for model_name in model_list:
            mdir = model_log_dir(model_name)
            epoch_log = CSVLogger(os.path.join(mdir, 'epoch_log.csv'), EPOCH_FIELDS)
            summary_log = CSVLogger(os.path.join(mdir, 'kfold_summary.csv'), KFOLD_SUMMARY_FIELDS)

            # Resume: skip completed folds + avoid epoch_log duplicates
            completed = set()
            existing_epochs = None
            if resume:
                completed = load_completed(
                    os.path.join(mdir, 'kfold_summary.csv'),
                    ['model', 'dataset', 'fold'])
                existing_epochs = load_completed_epochs(
                    os.path.join(mdir, 'epoch_log.csv'))

            batch_size = MODEL_CONFIG[model_name]['batch']
            print(f'\n>>> {model_name} x {dataset_name}')
            fold_accs, fold_f1s = [], []

            for fold_num, (train_idx, test_idx) in enumerate(folds):
                fold_key = (model_name, dataset_name, str(fold_num))
                if fold_key in completed:
                    print(f'  Fold {fold_num+1}/{N_FOLDS} [SKIP - already done]')
                    done_count += 1
                    continue

                set_seed(42 + fold_num)
                train_loader = DataLoader(
                    Subset(full_dataset, train_idx),
                    batch_size=batch_size, shuffle=True, num_workers=0)
                test_loader = DataLoader(
                    Subset(full_dataset, test_idx),
                    batch_size=batch_size, shuffle=False, num_workers=0)

                model = get_model(model_name, pretrained=True).to(device)
                t0 = time.time()

                result = train_loop(
                    model, train_loader, test_loader, device,
                    epoch_log, model_name, dataset_name, 'kfold', fold_num,
                    existing_epochs=existing_epochs)

                criterion = nn.CrossEntropyLoss()
                metrics = evaluate_full(model, test_loader, criterion, device)
                elapsed = time.time() - t0

                # Save best model weights per fold
                d_key = DS_MAP_REV[dataset_name]
                fold_weights_dir = f'outputs/weights_kfold/{d_key}'
                os.makedirs(fold_weights_dir, exist_ok=True)
                if result['best_state']:
                    torch.save(
                        {'model_state_dict': {k: v for k, v in result['best_state'].items()},
                         'best_epoch': result['best_epoch'],
                         'best_val_loss': result['best_val_loss']},
                        os.path.join(fold_weights_dir, f'best_{model_name}_fold{fold_num}.pth'))

                summary_log.write({
                    'model': model_name, 'dataset': dataset_name, 'fold': fold_num,
                    'best_epoch': result['best_epoch'],
                    'total_epochs': result['total_epochs'],
                    'final_lr': result['final_lr'],
                    'best_train_loss': result['best_train_loss'],
                    'best_train_acc': result['best_train_acc'],
                    'best_val_loss': result['best_val_loss'],
                    'best_val_acc': result['best_val_acc'],
                    'final_train_loss': result['final_train_loss'],
                    'final_train_acc': result['final_train_acc'],
                    'final_val_loss': result['final_val_loss'],
                    'final_val_acc': result['final_val_acc'],
                    'test_acc': round(metrics['acc'], 2),
                    'test_f1': round(metrics['f1'], 4),
                    'test_prec': round(metrics['prec'], 4),
                    'test_rec': round(metrics['rec'], 4),
                    'tp': metrics['tp'], 'fp': metrics['fp'],
                    'tn': metrics['tn'], 'fn': metrics['fn'],
                    'train_size': len(train_idx), 'test_size': len(test_idx),
                    'time_sec': round(elapsed, 1),
                })

                done_count += 1
                print(f'  Fold {fold_num+1}/{N_FOLDS}: '
                      f'Acc={metrics["acc"]:.2f}% F1={metrics["f1"]:.4f} '
                      f'(ep={result["best_epoch"]}/{result["total_epochs"]}, {elapsed:.0f}s) '
                      f'[{done_count}/{total}]')

                fold_accs.append(metrics['acc'])
                fold_f1s.append(metrics['f1'])

                del model
                torch.cuda.empty_cache()

            if fold_accs:
                print(f'  >> Mean: Acc={np.mean(fold_accs):.2f}±{np.std(fold_accs):.2f}%, '
                      f'F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}')

    # Consolidate per-model → merged CSVs + aggregation
    _consolidate_and_aggregate_kfold(model_list)


def _merge_per_model_csvs(model_list, filename, fieldnames):
    """Merge per-model CSVs into a single LOG_DIR/{merged_filename}."""
    merged_path = os.path.join(LOG_DIR, f'merged_{filename}')
    rows_all = []
    for model_name in model_list:
        src = os.path.join(model_log_dir(model_name), filename)
        if not os.path.exists(src):
            continue
        with open(src, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows_all.append(row)
    if not rows_all:
        return 0
    with open(merged_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_all)
    return len(rows_all)


def _consolidate_and_aggregate_kfold(model_list):
    """Merge per-model kfold CSVs → merged_epoch_log.csv, merged_kfold_summary.csv, merged_kfold_agg.csv."""
    print('\n--- Consolidating K-fold results ---')

    n_ep = _merge_per_model_csvs(model_list, 'epoch_log.csv', EPOCH_FIELDS)
    print(f'  merged_epoch_log.csv: {n_ep} rows')

    n_sum = _merge_per_model_csvs(model_list, 'kfold_summary.csv', KFOLD_SUMMARY_FIELDS)
    print(f'  merged_kfold_summary.csv: {n_sum} rows')

    # Aggregation from merged summary
    merged_summary = os.path.join(LOG_DIR, 'merged_kfold_summary.csv')
    if not os.path.exists(merged_summary):
        return

    from collections import defaultdict
    data = defaultdict(list)
    with open(merged_summary, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['dataset'])
            data[key].append(row)

    agg_fields = [
        'model', 'dataset', 'feature_map_res',
        'acc_mean', 'acc_std', 'f1_mean', 'f1_std',
        'fold1_acc', 'fold2_acc', 'fold3_acc',
        'fold1_f1', 'fold2_f1', 'fold3_f1',
        'avg_best_epoch', 'avg_total_epochs',
    ]
    agg_path = os.path.join(LOG_DIR, 'merged_kfold_agg.csv')
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for (model, dataset), rows in sorted(data.items()):
            if len(rows) < N_FOLDS:
                continue
            rows_sorted = sorted(rows, key=lambda r: int(r['fold']))
            accs = [float(r['test_acc']) for r in rows_sorted]
            f1s = [float(r['test_f1']) for r in rows_sorted]
            best_eps = [int(r['best_epoch']) for r in rows_sorted]
            total_eps = [int(r['total_epochs']) for r in rows_sorted]
            res = MODEL_CONFIG.get(model, {}).get('res', '?')
            writer.writerow({
                'model': model, 'dataset': dataset, 'feature_map_res': res,
                'acc_mean': round(np.mean(accs), 2),
                'acc_std': round(np.std(accs), 2),
                'f1_mean': round(np.mean(f1s), 4),
                'f1_std': round(np.std(f1s), 4),
                'fold1_acc': round(accs[0], 2),
                'fold2_acc': round(accs[1], 2),
                'fold3_acc': round(accs[2], 2),
                'fold1_f1': round(f1s[0], 4),
                'fold2_f1': round(f1s[1], 4),
                'fold3_f1': round(f1s[2], 4),
                'avg_best_epoch': round(np.mean(best_eps), 1),
                'avg_total_epochs': round(np.mean(total_eps), 1),
            })
    print(f'  merged_kfold_agg.csv → {agg_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Single-split Training + Grad-CAM++
# ═══════════════════════════════════════════════════════════════════════════════
class GradCAMPlusPlus:
    """Grad-CAM++ (Chattopadhay et al., 2018) with second-order gradient weights."""
    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.activations = None
        self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, module, input, output):
        self.activations = output.clone().detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        score = output[0, target_class]
        self.model.zero_grad()
        score.backward()

        if self.activations is None or self.gradients is None:
            return None

        grads = self.gradients          # (1, C, H, W)
        acts = self.activations         # (1, C, H, W)
        b, k, u, v = grads.size()

        # Grad-CAM++ alpha weights (second-order)
        alpha_num = grads.pow(2)
        alpha_denom = grads.pow(2).mul(2) + \
            acts.mul(grads.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)

        positive_gradients = F.relu(score.exp().detach() * grads)
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()


def phase2_gradcam(model_list, device, resume=False):
    print('\n' + '=' * 70)
    print('PHASE 2: Single-split Train + Grad-CAM++ on TEST SET')
    total = len(model_list) * len(DATASETS)
    print(f'  Models: {len(model_list)} | Datasets: {len(DATASETS)}')
    print(f'  Total: {total} runs')
    print('=' * 70)

    done_count = 0
    for dataset_name in DATASETS:
        for model_name in model_list:
            mdir = model_log_dir(model_name)
            epoch_log = CSVLogger(os.path.join(mdir, 'epoch_log.csv'), EPOCH_FIELDS)
            summary_log = CSVLogger(
                os.path.join(mdir, 'singlesplit_summary.csv'), SINGLESPLIT_SUMMARY_FIELDS)

            # Resume checks
            completed = set()
            existing_epochs = None
            if resume:
                completed = load_completed(
                    os.path.join(mdir, 'singlesplit_summary.csv'),
                    ['model', 'dataset'])
                existing_epochs = load_completed_epochs(
                    os.path.join(mdir, 'epoch_log.csv'))

            run_key = (model_name, dataset_name)
            if run_key in completed:
                print(f'\n>>> {model_name} x {dataset_name} [SKIP]')
                done_count += 1
                continue

            batch_size = MODEL_CONFIG[model_name]['batch']
            set_seed(SEED)
            transform = transforms.Compose([transforms.ToTensor()])
            full_dataset = ImageFolder(
                root=os.path.join(DATA_ROOT, dataset_name), transform=transform)

            n = len(full_dataset)
            train_size = int(TRAIN_RATIO * n)
            val_size = int(VAL_RATIO * n)
            test_size = n - train_size - val_size
            train_ds, val_ds, test_ds = random_split(
                full_dataset, [train_size, val_size, test_size])

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            model = get_model(model_name, pretrained=True).to(device)
            print(f'\n>>> {model_name} x {dataset_name} '
                  f'(train={train_size}, val={val_size}, test={test_size})')
            t0 = time.time()

            result = train_loop(
                model, train_loader, val_loader, device,
                epoch_log, model_name, dataset_name, 'singlesplit', -1,
                existing_epochs=existing_epochs)

            # Save best + last weights
            d_key = DS_MAP_REV[dataset_name]
            weights_dir = f'outputs/weights/{d_key}'
            os.makedirs(weights_dir, exist_ok=True)
            torch.save(
                {'model_state_dict': model.state_dict(),
                 'best_epoch': result['best_epoch'],
                 'best_val_loss': result['best_val_loss']},
                os.path.join(weights_dir, f'best_{model_name}.pth'))
            if result['last_state']:
                torch.save(
                    {'model_state_dict': {k: v for k, v in result['last_state'].items()},
                     'final_epoch': result['total_epochs'],
                     'final_val_loss': result['final_val_loss']},
                    os.path.join(weights_dir, f'last_{model_name}.pth'))

            # Evaluate on TEST set
            criterion = nn.CrossEntropyLoss()
            test_metrics = evaluate_full(model, test_loader, criterion, device)

            # Inference latency (per-sample, P1-P99 trimmed)
            infer_stats = measure_inference_latency(model, test_loader, device)
            print(f'    test_acc={test_metrics["acc"]:.2f}% '
                  f'F1={test_metrics["f1"]:.4f} '
                  f'(ep={result["best_epoch"]}/{result["total_epochs"]})')
            print(f'    latency: mean={infer_stats["infer_mean_ms"]:.2f}ms '
                  f'median={infer_stats["infer_median_ms"]:.2f}ms '
                  f'P1={infer_stats["infer_p1_ms"]:.2f} P99={infer_stats["infer_p99_ms"]:.2f}ms '
                  f'(n={infer_stats["infer_n_samples"]})')

            # ── Grad-CAM++ on TEST SET ──
            model = disable_inplace_relu(model)
            model.eval()
            target_layer = get_target_layer(model, model_name)
            cam_gen = GradCAMPlusPlus(model, target_layer, device)
            class_names = full_dataset.classes  # ['bee', 'mite']

            output_dir = os.path.join('outputs/gradcam', d_key, model_name)
            os.makedirs(output_dir, exist_ok=True)

            n_cam = 0
            for idx in test_ds.indices:
                img_path, label = full_dataset.samples[idx]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                true_class = class_names[label]
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor.to(device))
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_class = int(np.argmax(probs))

                for cls_idx, cls_name in enumerate(class_names):
                    cam = cam_gen.generate(input_tensor, cls_idx)
                    if cam is not None:
                        np.save(os.path.join(
                            output_dir, f'{img_name}_{cls_name}_cam_data.npy'), cam)

                metadata = {
                    'image': os.path.basename(img_path),
                    'true_class': true_class, 'true_label': label,
                    'model': model_name, 'dataset': dataset_name,
                    'data_split': 'test',
                    'prediction': {
                        'class_name': class_names[pred_class],
                        'confidence': float(probs[pred_class]),
                        'correct': pred_class == label,
                    },
                }
                cam_path = os.path.join(output_dir, f'{img_name}_mite_cam_data.npy')
                if os.path.exists(cam_path):
                    metadata['cam_shape'] = list(np.load(cam_path).shape)
                with open(os.path.join(
                        output_dir, f'{img_name}_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                n_cam += 1

            elapsed = time.time() - t0
            done_count += 1

            summary_log.write({
                'model': model_name, 'dataset': dataset_name,
                'best_epoch': result['best_epoch'],
                'total_epochs': result['total_epochs'],
                'final_lr': result['final_lr'],
                'best_train_loss': result['best_train_loss'],
                'best_train_acc': result['best_train_acc'],
                'best_val_loss': result['best_val_loss'],
                'best_val_acc': result['best_val_acc'],
                'final_train_loss': result['final_train_loss'],
                'final_train_acc': result['final_train_acc'],
                'final_val_loss': result['final_val_loss'],
                'final_val_acc': result['final_val_acc'],
                'test_acc': round(test_metrics['acc'], 2),
                'test_f1': round(test_metrics['f1'], 4),
                'test_prec': round(test_metrics['prec'], 4),
                'test_rec': round(test_metrics['rec'], 4),
                'tp': test_metrics['tp'], 'fp': test_metrics['fp'],
                'tn': test_metrics['tn'], 'fn': test_metrics['fn'],
                **infer_stats,
                'train_size': train_size, 'val_size': val_size,
                'test_size': test_size, 'n_gradcam': n_cam,
                'time_sec': round(elapsed, 1),
            })

            print(f'    Grad-CAM: {n_cam} images → {output_dir} '
                  f'({elapsed:.0f}s) [{done_count}/{total}]')

            cam_gen.remove_hooks()
            del model, cam_gen
            torch.cuda.empty_cache()

    # Consolidate per-model → merged CSVs
    _consolidate_singlesplit(model_list)


def _consolidate_singlesplit(model_list):
    """Merge per-model singlesplit CSVs → merged_singlesplit_summary.csv."""
    print('\n--- Consolidating single-split results ---')
    # epoch_log already merged by kfold consolidation; re-merge to include singlesplit epochs
    n_ep = _merge_per_model_csvs(model_list, 'epoch_log.csv', EPOCH_FIELDS)
    print(f'  merged_epoch_log.csv: {n_ep} rows (kfold + singlesplit)')

    n_sum = _merge_per_model_csvs(model_list, 'singlesplit_summary.csv', SINGLESPLIT_SUMMARY_FIELDS)
    print(f'  merged_singlesplit_summary.csv: {n_sum} rows')


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: XAI Metrics (Annotation-free & Localization)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Annotation-free metrics ---

def _activation_coverage(cam, threshold=0.3):
    return float(np.mean(cam >= threshold))

def _heatmap_entropy(cam, n_bins=50):
    hist, _ = np.histogram(cam.flatten(), bins=n_bins, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))

def _peak_concentration(cam, top_percent=0.1):
    total = np.sum(cam)
    if total == 0:
        return 0.0
    thr = np.quantile(cam, 1 - top_percent)
    return float(np.sum(cam[cam >= thr]) / total)

def _spatial_compactness(cam, threshold=0.3):
    from scipy import ndimage
    binary = (cam >= threshold).astype(np.float32)
    if binary.sum() == 0:
        return 0.0
    labeled, n_features = ndimage.label(binary)
    if n_features == 0:
        return 0.0
    sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
    return float(max(sizes) / binary.sum())

def _energy_ratio(cam, top_k=0.2):
    total = np.sum(cam)
    if total == 0:
        return 0.0
    n_top = max(1, int(cam.size * top_k))
    sorted_vals = np.sort(cam.flatten())[::-1]
    return float(np.sum(sorted_vals[:n_top]) / total)

# --- Localization metrics (require bounding-box annotations) ---

def _get_resize_type(dataset_name):
    if '_resized_set' in dataset_name:
        return 'preserve'         # _set = MR (aspect-ratio preserving + padding)
    elif '_resized' in dataset_name:
        return 'stretch'          # _resized only = NR (stretch to 224x224)
    return 'none'

def _transform_bbox_224(bbox, orig_w, orig_h, resize_type, target=224):
    x1, y1, x2, y2 = bbox
    if resize_type == 'preserve':
        scale = target / max(orig_w, orig_h)  # always scale — matches resize.py
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_left, pad_top = (target - new_w) // 2, (target - new_h) // 2
        nx1, ny1 = pad_left + x1 * scale, pad_top + y1 * scale
        nx2, ny2 = pad_left + x2 * scale, pad_top + y2 * scale
    elif resize_type == 'stretch':
        sx, sy = target / orig_w, target / orig_h
        nx1, ny1, nx2, ny2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    else:  # none: original size centered with padding, NO scaling
        pad_left, pad_top = (target - orig_w) // 2, (target - orig_h) // 2
        nx1, ny1 = pad_left + x1, pad_top + y1
        nx2, ny2 = pad_left + x2, pad_top + y2
    return tuple(max(0, min(target, v)) for v in (nx1, ny1, nx2, ny2))

def _cam_resize_normalize(cam, target=224):
    import cv2
    if cam.shape[0] != target or cam.shape[1] != target:
        cam = cv2.resize(cam, (target, target), interpolation=cv2.INTER_LINEAR)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

def _iou(cam_mask, bbox, target=224):
    bbox_mask = np.zeros((target, target), dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(target, x2), min(target, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    bbox_mask[y1:y2, x1:x2] = 1.0
    inter = np.logical_and(cam_mask, bbox_mask).sum()
    union = np.logical_or(cam_mask, bbox_mask).sum()
    return float(inter / union) if union > 0 else 0.0

def _pointing_game(cam, bbox):
    my, mx = np.unravel_index(np.argmax(cam), cam.shape)
    x1, y1, x2, y2 = bbox
    return 1.0 if (x1 <= mx <= x2 and y1 <= my <= y2) else 0.0

def _distance_error(cam, bbox, target=224):
    my, mx = np.unravel_index(np.argmax(cam), cam.shape)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return float(np.sqrt((mx - cx)**2 + (my - cy)**2) / (np.sqrt(2) * target))

def _energy_inside(cam, bbox, target=224):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(target, x2), min(target, y2)
    total = cam.sum()
    if total == 0:
        return 0.0
    return float(cam[y1:y2, x1:x2].sum() / total)


def phase3_xai(model_list):
    import pandas as pd
    print('\n' + '=' * 70)
    print('PHASE 3: XAI Metrics (annotation-free + localization)')
    print('=' * 70)

    gradcam_dir = Path('outputs/gradcam')

    # ── 3a. Annotation-free metrics ──
    print('\n--- Annotation-free metrics ---')
    annfree_results = []

    for dk, dataset_name in DS_MAP.items():
        for model_name in model_list:
            model_dir = gradcam_dir / dk / model_name
            if not model_dir.exists():
                continue
            res = MODEL_CONFIG[model_name]['res']

            for npy_file in sorted(model_dir.glob('*_cam_data.npy')):
                parts = npy_file.name.replace('_cam_data.npy', '').rsplit('_', 1)
                if len(parts) != 2:
                    continue
                image_id, target_class = parts
                try:
                    cam = np.load(npy_file)
                    if cam.max() > 1.0:
                        cam = cam / (cam.max() + 1e-10)
                    if cam.ndim == 3:
                        cam = cam.squeeze()
                    if cam.ndim != 2:
                        continue
                    annfree_results.append({
                        'dataset': dk, 'dataset_name': dataset_name,
                        'model': model_name, 'image_id': image_id,
                        'target_class': target_class,
                        'feature_map_res': res,
                        'coverage_30': _activation_coverage(cam, 0.3),
                        'coverage_50': _activation_coverage(cam, 0.5),
                        'entropy': _heatmap_entropy(cam),
                        'peak_concentration_10': _peak_concentration(cam, 0.10),
                        'energy_ratio_20': _energy_ratio(cam, 0.20),
                        'spatial_compactness_30': _spatial_compactness(cam, 0.30),
                    })
                except Exception:
                    pass

    if annfree_results:
        df = pd.DataFrame(annfree_results)
        out_path = os.path.join(LOG_DIR, 'xai_annfree.csv')
        df.to_csv(out_path, index=False)
        print(f'  {len(df)} records → {out_path}')

        mite_df = df[df['target_class'] == 'mite'] if 'mite' in df['target_class'].values else df
        print(f'\n  {"Model":30s} {"Cov@30":>8s} {"Entropy":>8s} '
              f'{"PC@10":>8s} {"SC@30":>8s} {"Res":>6s}')
        print('  ' + '-' * 75)
        for model_name in model_list:
            sub = mite_df[mite_df['model'] == model_name]
            if len(sub) == 0:
                continue
            res = MODEL_CONFIG[model_name]['res']
            print(f'  {model_name:30s} '
                  f'{sub["coverage_30"].mean():8.4f} '
                  f'{sub["entropy"].mean():8.4f} '
                  f'{sub["peak_concentration_10"].mean():8.4f} '
                  f'{sub["spatial_compactness_30"].mean():8.4f} '
                  f'{res:>6s}')
    else:
        print('  No Grad-CAM data found. Run phase 2 (gradcam) first.')

    # ── 3b. Localization metrics ──
    print('\n--- Localization metrics ---')
    bbox_csv = Path('mite_bbox_annotations.csv')
    if not bbox_csv.exists():
        print(f'  [SKIP] YOLO bbox file not found: {bbox_csv}')
        return

    bbox_best = {}
    with open(bbox_csv) as f:
        for r in csv.DictReader(f):
            fn = r['filename']
            det = {
                'bbox': (float(r['mite_x1']), float(r['mite_y1']),
                         float(r['mite_x2']), float(r['mite_y2'])),
                'orig_w': int(r['orig_w']), 'orig_h': int(r['orig_h']),
                'conf': float(r['confidence']),
            }
            if fn not in bbox_best or det['conf'] > bbox_best[fn]['conf']:
                bbox_best[fn] = det

    loc_results = []
    for dk, dataset_name in DS_MAP.items():
        resize_type = _get_resize_type(dataset_name)
        for model_name in model_list:
            model_dir = gradcam_dir / dk / model_name
            if not model_dir.exists():
                continue
            res = MODEL_CONFIG[model_name]['res']

            for cam_path in sorted(model_dir.glob('*_mite_cam_data.npy')):
                img_name = cam_path.name.replace('_mite_cam_data.npy', '')
                img_file = img_name + '.png'
                if img_file not in bbox_best:
                    continue
                cam_raw = np.load(cam_path)
                if cam_raw.max() == 0:
                    continue

                bd = bbox_best[img_file]
                bbox_224 = _transform_bbox_224(
                    bd['bbox'], bd['orig_w'], bd['orig_h'], resize_type)
                cam_224 = _cam_resize_normalize(cam_raw)
                cam_mask_50 = (cam_224 >= 0.5).astype(np.float32)
                cam_mask_30 = (cam_224 >= 0.3).astype(np.float32)

                loc_results.append({
                    'dataset': dataset_name, 'dataset_id': dk,
                    'model': model_name, 'image': img_name,
                    'feature_map_res': res,
                    'iou_t50': round(_iou(cam_mask_50, bbox_224), 4),
                    'iou_t30': round(_iou(cam_mask_30, bbox_224), 4),
                    'pointing_game': int(_pointing_game(cam_224, bbox_224)),
                    'distance_norm': round(_distance_error(cam_224, bbox_224), 4),
                    'energy_inside': round(_energy_inside(cam_224, bbox_224), 4),
                })

    if loc_results:
        ldf = pd.DataFrame(loc_results)
        out_path = os.path.join(LOG_DIR, 'xai_localization.csv')
        ldf.to_csv(out_path, index=False)
        print(f'  {len(ldf)} records → {out_path}')

        print(f'\n  {"Model":30s} {"IoU@50":>8s} {"IoU@30":>8s} '
              f'{"PG":>6s} {"Dist":>8s} {"Energy":>8s} {"N":>5s}')
        print('  ' + '-' * 80)
        for model_name in model_list:
            sub = ldf[ldf['model'] == model_name]
            if len(sub) == 0:
                continue
            print(f'  {model_name:30s} '
                  f'{sub["iou_t50"].mean():8.4f} '
                  f'{sub["iou_t30"].mean():8.4f} '
                  f'{sub["pointing_game"].mean():6.3f} '
                  f'{sub["distance_norm"].mean():8.4f} '
                  f'{sub["energy_inside"].mean():8.4f} '
                  f'{len(sub):5d}')

        # Resolution comparison
        print(f'\n  --- Resolution vs Localization ---')
        for res in ['7x7', '14x14', '28x28', '56x56']:
            sub = ldf[ldf['feature_map_res'] == res]
            if len(sub) == 0:
                continue
            print(f'  {res:>6s}  IoU@30={sub["iou_t30"].mean():.4f}  '
                  f'PG={sub["pointing_game"].mean():.3f}  '
                  f'Energy={sub["energy_inside"].mean():.4f}  '
                  f'(n={len(sub)})')

        # Kruskal-Wallis
        from scipy.stats import kruskal
        groups = {}
        for res in ['7x7', '14x14', '28x28', '56x56']:
            vals = ldf[ldf['feature_map_res'] == res]['iou_t30'].values
            if len(vals) > 0:
                groups[res] = vals
        if len(groups) >= 2:
            h, p = kruskal(*groups.values())
            print(f'\n  Kruskal-Wallis (IoU@30): H={h:.4f}, p={p:.6f}')
    else:
        print('  No localization data found.')


# ═══════════════════════════════════════════════════════════════════════════════
# Model Metadata
# ═══════════════════════════════════════════════════════════════════════════════
def save_model_metadata(model_list):
    """Save parameter counts and architecture info for all models."""
    meta_path = os.path.join(LOG_DIR, 'model_metadata.csv')
    os.makedirs(LOG_DIR, exist_ok=True)

    fields = ['model', 'feature_map_res', 'attention', 'architecture',
              'total_params', 'trainable_params', 'batch_size']
    rows = []
    for model_name in model_list:
        cfg = MODEL_CONFIG[model_name]
        model = get_model(model_name, pretrained=False)
        total, trainable = count_params(model)
        rows.append({
            'model': model_name,
            'feature_map_res': cfg['res'],
            'attention': cfg['attn'],
            'architecture': cfg['arch'],
            'total_params': total,
            'trainable_params': trainable,
            'batch_size': cfg['batch'],
        })
        del model

    with open(meta_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nModel metadata → {meta_path}')
    for r in rows:
        print(f'  {r["model"]:30s}  {r["feature_map_res"]:>6s}  '
              f'{r["total_params"]:>10,d} params  '
              f'attn={r["attention"]}')


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture Verification
# ═══════════════════════════════════════════════════════════════════════════════
def verify_architectures(model_list):
    """Verify feature map sizes for all models."""
    print('\n[Architecture Verification]')
    x = torch.randn(1, 3, 224, 224)
    all_ok = True

    for model_name in model_list:
        cfg = MODEL_CONFIG[model_name]
        expected_res = cfg['res']
        expected_h = int(expected_res.split('x')[0])

        model = get_model(model_name, pretrained=False)
        model.eval()
        target_layer = get_target_layer(model, model_name)

        feat_shape = {}
        def hook_fn(module, inp, out, name=model_name):
            feat_shape[name] = out.shape
        handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(x)
        handle.remove()

        shape = feat_shape[model_name]
        h, w = shape[2], shape[3]
        ok = (h == expected_h and w == expected_h)
        status = 'OK' if ok else f'FAIL (got {h}x{w})'
        if not ok:
            all_ok = False
        print(f'  {model_name:30s} → {h}x{w} (expected {expected_res}) [{status}]')
        del model

    del x
    if not all_ok:
        print('\n  [ERROR] Some models failed verification!')
        sys.exit(1)
    print('  All models verified.\n')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Unified 21-model experiment with comprehensive training logs')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index (0 or 1). Omit for auto-select.')
    parser.add_argument('--phase', nargs='+',
                        choices=['kfold', 'gradcam', 'xai', 'metadata', 'verify'],
                        default=['kfold', 'gradcam', 'xai'],
                        help='Phases to run (default: all three)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific model names. Default: all 21.')
    parser.add_argument('--resolution', nargs='+',
                        choices=['7x7', '14x14', '28x28', '56x56'],
                        default=None,
                        help='Filter models by resolution group.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed runs (checks summary CSVs).')
    parser.add_argument('--list-models', action='store_true',
                        help='Print all model names and exit.')
    args = parser.parse_args()

    if args.list_models:
        print('Available models (21):')
        for res in ['7x7', '14x14', '28x28', '56x56']:
            names = [m for m, c in MODEL_CONFIG.items() if c['res'] == res]
            print(f'\n  {res} ({len(names)}):')
            for n in names:
                c = MODEL_CONFIG[n]
                print(f'    {n:30s}  batch={c["batch"]:2d}  attn={c["attn"]}')
        return

    # Resolve model list
    if args.models:
        model_list = []
        for m in args.models:
            if m not in MODEL_CONFIG:
                print(f'Unknown model: {m}')
                sys.exit(1)
            model_list.append(m)
    elif args.resolution:
        model_list = [m for m, c in MODEL_CONFIG.items()
                      if c['res'] in args.resolution]
    else:
        model_list = list(MODEL_CONFIG.keys())

    # Device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        print(f'[GPU] {torch.cuda.get_device_name(device)}')
    else:
        print('[WARNING] CPU mode — training will be very slow.')

    print(f'\nModels ({len(model_list)}): {model_list}')
    print(f'Datasets: {len(DATASETS)}')
    print(f'Phases: {args.phase}')
    print(f'Resume: {args.resume}')
    print(f'Log directory: {LOG_DIR}')
    os.makedirs(LOG_DIR, exist_ok=True)

    # Always save metadata and verify
    if 'verify' in args.phase or 'kfold' in args.phase or 'gradcam' in args.phase:
        verify_architectures(model_list)
    if 'metadata' in args.phase or 'kfold' in args.phase or 'gradcam' in args.phase:
        save_model_metadata(model_list)

    if 'kfold' in args.phase:
        phase1_kfold(model_list, device, args.resume)

    if 'gradcam' in args.phase:
        phase2_gradcam(model_list, device, args.resume)

    if 'xai' in args.phase:
        phase3_xai(model_list)

    print('\n' + '=' * 70)
    print('COMPLETE')
    print(f'\n  [Per-model logs]')
    print(f'    {LOG_DIR}/{{model}}/epoch_log.csv')
    print(f'    {LOG_DIR}/{{model}}/kfold_summary.csv')
    print(f'    {LOG_DIR}/{{model}}/singlesplit_summary.csv')
    print(f'\n  [Merged (all models)]')
    print(f'    {LOG_DIR}/merged_epoch_log.csv')
    print(f'    {LOG_DIR}/merged_kfold_summary.csv')
    print(f'    {LOG_DIR}/merged_kfold_agg.csv')
    print(f'    {LOG_DIR}/merged_singlesplit_summary.csv')
    print(f'    {LOG_DIR}/xai_annfree.csv')
    print(f'    {LOG_DIR}/xai_localization.csv')
    print(f'    {LOG_DIR}/model_metadata.csv')
    print(f'\n  [Weights]')
    print(f'    outputs/weights/d*/best_{{model}}.pth')
    print(f'    outputs/weights/d*/last_{{model}}.pth')
    print(f'    outputs/weights_kfold/d*/best_{{model}}_fold{{n}}.pth')
    print(f'\n  [Grad-CAM++]')
    print(f'    outputs/gradcam/d*/{{model}}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
