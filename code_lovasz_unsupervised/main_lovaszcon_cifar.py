from __future__ import print_function

import os
from random import choices
import sys
import argparse
import time
import math
import numpy as np

from typing import List

import torch
import torchvision
import torch.backends.cudnn as cudnn

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv import transforms
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, NDArrayDecoder
from ffcv_extra.transforms import RandomGrayscale, MultiViewRandomHorizontalFlip, MultiViewToTorchImage
from ffcv_extra.decoders import MultiViewRandomResizedCropRGBImageDecoder
from networks.resnet_big import SupConResNet

from util import AverageMeter, TwoCropTransform
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from losses import SupConLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--loader_type', type=str, default='classic', choices=['classic', 'ffcv'], help='Dataloader type.')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data', type=str, default=None, help='FFCV format file for data.')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='Lovasz',
                        choices=['SupCE', 'Lovasz', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--stable', action='store_true',
                        help='numerical stability term')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to model used as base start')
    parser.add_argument('--sim_type', type=str, default=None,
                        choices=['adhoc', 'projection', 'dummy', 'SupCon', 'CE', 'CLIP', 'topk', 'superclass', 'superclass08'], help='type of matrix of class similarities')
    parser.add_argument('--projection_dim', type=int, default=100,
                        help='Projection dimension.')

    parser.add_argument('--pretrained', action='store_true', help='Pretrained ResNet?')

    parser.add_argument('--blend', type=float, default=None,
                        help='Blending.')

    opt = parser.parse_args()

    
    opt.model_path = './save/SupCon/cifar100_models'
    opt.tb_path = './save/SupCon/cifar100_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, 'cifar100', opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.dataset = 'cifar100'

    if opt.sim_type == 'SupCon':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_matrix.csv')
    elif opt.sim_type == 'superclass':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_superclass.csv')
    elif opt.sim_type == 'superclass08':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_superclass_08.csv')
    elif opt.sim_type == 'CLIP':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_graph_clip.csv')
    elif opt.sim_type == 'CE':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_matrix_CE.csv')
    elif opt.sim_type == 'topk':
        opt.sim_mat = os.path.join('./similarities', opt.dataset, 'similarity_matrix_topk.csv')
    elif opt.sim_type == 'dummy':
        opt.sim_mat = 'dummy'
    else:
        opt.sim_mat = None

    opt.model_name = '{0:}_simtype_{1:}'.format(opt.model_name, opt.sim_type)
    
    if opt.stable:
        opt.model_name = '{}_stable'.format(opt.model_name)

    if opt.blend is not None:
        opt.model_name = '{0:}_blend_{1:0.2f}'.format(opt.model_name, opt.blend)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    if opt.loader_type == 'ffcv':
        # construct data loader
        mean = np.array([0.5071, 0.4867, 0.4408]) * 255
        std = np.array([0.2675, 0.2565, 0.2761]) * 255

        normalize = transforms.NormalizeImage(mean=np.array(mean), std=np.array(std), type=np.dtype(np.float32))

        image_pipeline: List[Operation] = [
            MultiViewRandomResizedCropRGBImageDecoder(output_size=(opt.size, opt.size), num_views=2, scale=(0.2, 1.)),
            MultiViewRandomHorizontalFlip(),
            transforms.RandomBrightness(0.4),
            transforms.RandomContrast(0.4),
            transforms.RandomSaturation(0.4),
            RandomGrayscale(p=0.2, multi_view=True),
            normalize,
            transforms.ToTensor(),
            MultiViewToTorchImage(),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            transforms.ToTensor(),
            transforms.Squeeze(),
        ]

        embed_pipeline: List[Operation] = [
            NDArrayDecoder(),
            transforms.ToTensor(),
        ]

        train_loader = Loader(
            opt.data,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={
                'image': image_pipeline,
                'clip_embed': embed_pipeline,
                'label': label_pipeline,
            },
            distributed=False
        )
    
    else:
        # construct data loader
        if opt.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif opt.dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))
        normalize = torchvision.transforms.Normalize(mean=mean, std=std)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # type: ignore
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        if opt.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root=opt.data,
                                            transform=TwoCropTransform(train_transform),
                                            download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(root=opt.data,
                                            transform=TwoCropTransform(train_transform),
                                            download=True)
        else:
            raise ValueError(opt.dataset)

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(  # type: ignore
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):

    if opt.sim_mat is not None and opt.sim_mat != 'dummy':
        sim_mat = torch.from_numpy(np.loadtxt(opt.sim_mat, delimiter=','))
    elif opt.sim_mat == 'dummy':
        sim_mat = torch.eye(100)
    else:
        sim_mat = None
    
    model = SupConResNet(name=opt.model)
    if opt.sim_type == 'projection':
        model.clip_projection = torch.nn.Linear(1024, opt.projection_dim)

    if opt.pretrained:
        pretrained_model = getattr(torchvision.models, opt.model)(weights='IMAGENET1K_V1')
        state_dict = pretrained_model.state_dict()
        new_state_dict = model.state_dict()
        encoder_state_dict = model.encoder.state_dict()
        for key in state_dict.keys():
            if key in encoder_state_dict.keys() and not 'conv1' in key:
                new_state_dict['encoder.'+key] = state_dict[key]
        model.load_state_dict(new_state_dict)
    
    if opt.method != 'SupCE':
        criterion = SupConLoss(temperature=opt.temp, sim_mat=sim_mat, stability=opt.stable, blend=opt.blend)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.ckpt is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.parallel.DataParallel(model.encoder)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True
            
    else:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.parallel.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

            model.load_state_dict(state_dict)

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for idx, (images, embeds, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        if opt.loader_type == 'ffcv':
            images = torch.cat(torch.unbind(images, dim=1), dim=0)
        else:
            images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            embeds = embeds.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # 2 views per image
        embeds = torch.cat([embeds, embeds], dim=0)
        
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'Lovasz':
            if opt.sim_type == 'adhoc':
                with torch.no_grad():
                    sample_sim = embeds / torch.linalg.norm(embeds, dim=-1, keepdim=True)
                    sample_sim_mat = sample_sim @ sample_sim.T
                loss = criterion(features, sample_sim_mat = sample_sim_mat)
            elif opt.sim_type == 'projection':
                sample_sim = model.clip_projection(embeds)
                sample_sim = sample_sim / torch.linalg.norm(sample_sim, dim=-1, keepdim=True)
                sample_sim_mat = sample_sim @ sample_sim.T
                loss = criterion(features, sample_sim_mat = sample_sim_mat)
            else:
                loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)


    # build optimizer
    optimizers = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizers, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizers, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizers, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizers, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()

