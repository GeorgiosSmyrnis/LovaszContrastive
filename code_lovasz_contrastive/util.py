from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import itertools

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizers, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    if not hasattr(optimizers, '__iter__'):
        optimizers_list = [optimizers]
    else:
        optimizers_list = optimizers

    for optimizer in optimizers_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizers):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for optimizer in [optimizers]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def get_elem_gen_tuple(gen, idx):
    while True:
        try:
            item = next(gen)
            yield item[idx]
        except StopIteration:
            return

def is_slow_param(named_param):
    name, _ = named_param
    return (name == 'low_val') or (name == 'high_val')

def set_optimizer(opt, model, pair=False):
    if pair:
        optimizer = optim.SGD(get_elem_gen_tuple(itertools.filterfalse(is_slow_param, model.named_parameters()), 1),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
                          
        optimizer_slow = optim.SGD(get_elem_gen_tuple(filter(is_slow_param, model.named_parameters()), 1),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
        return optimizer, optimizer_slow

    else:
        
        optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

        return optimizer

def avg_grad(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(torch.linalg.norm(p.grad).item())
    return np.mean(norms)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
