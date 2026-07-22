import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_single_loss(pred, pngs, labels, weights, num_classes, dice_loss, focal_loss, dice_weight=1.0, label_smoothing=0.0):
    if focal_loss:
        loss = Focal_Loss(pred, pngs, weights, num_classes=num_classes)
    else:
        loss = CE_Loss(pred, pngs, weights, num_classes=num_classes, label_smoothing=label_smoothing)

    if dice_loss:
        main_dice = Dice_loss(pred, labels)
        loss = loss + dice_weight * main_dice

    return loss

def Boundary_Loss(boundary_logits, target, ignore_index=None, smooth=1e-5):
    if boundary_logits.size(1) > 1:
        boundary_logits = boundary_logits[:, 1:2]

    n, c, h, w = boundary_logits.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        boundary_logits = F.interpolate(boundary_logits, size=(ht, wt), mode="bilinear", align_corners=True)

    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index

    fg = ((target > 0) & valid).float().unsqueeze(1)
    boundary = torch.zeros_like(fg)
    boundary[:, :, :, 1:] = torch.maximum(boundary[:, :, :, 1:], (fg[:, :, :, 1:] - fg[:, :, :, :-1]).abs())
    boundary[:, :, :, :-1] = torch.maximum(boundary[:, :, :, :-1], (fg[:, :, :, 1:] - fg[:, :, :, :-1]).abs())
    boundary[:, :, 1:, :] = torch.maximum(boundary[:, :, 1:, :], (fg[:, :, 1:, :] - fg[:, :, :-1, :]).abs())
    boundary[:, :, :-1, :] = torch.maximum(boundary[:, :, :-1, :], (fg[:, :, 1:, :] - fg[:, :, :-1, :]).abs())
    boundary = boundary * valid.float().unsqueeze(1)

    pos = boundary.sum()
    neg = boundary.numel() - pos
    pos_weight = (neg / (pos + smooth)).clamp(1.0, 20.0).detach()
    bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary, pos_weight=pos_weight)

    prob = torch.sigmoid(boundary_logits)
    prob = prob * valid.float().unsqueeze(1)
    inter = (prob * boundary).sum()
    dice = 1 - (2 * inter + smooth) / (prob.sum() + boundary.sum() + smooth)
    return bce + dice

def CE_Loss(inputs, target, cls_weights, num_classes=21, label_smoothing=0.0):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(
        weight=cls_weights,
        ignore_index=num_classes,
        label_smoothing=label_smoothing,
    )(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    def poly_lr(lr, min_lr, total_iters, power, iters):
        iters = min(iters, total_iters)
        factor = pow(1 - iters / float(total_iters), power)
        return min_lr + (lr - min_lr) * factor

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "poly":
        func = partial(poly_lr, lr, min_lr, total_iters, 2.0)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_mult', 1.0)
