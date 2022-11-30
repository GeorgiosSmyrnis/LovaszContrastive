from typing import List

import sys
import argparse
import time
import math
import numpy as np
import os

import torch
import torchvision
import torch.backends.cudnn as cudnn

from confusion_matrix_similarity import get_similarity

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv import transforms
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier


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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--train_data', type=str, default=None, help='FFCV format file for data (train).')
    parser.add_argument('--val_data', type=str, default=None, help='FFCV format file for data (val).')
    parser.add_argument('--train_size', type=int, default=32, help='parameter for RandomResizedCrop (train)')
    parser.add_argument('--val_size', type=int, default=32, help='parameter for RandomResizedCrop (val)')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    parser.add_argument('--save_similarity', action='store_true',
                        help='save the similarity graph of classes')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format('imagenet', opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    opt.n_cls = 100

    return opt

def set_loader(opt):
    mean = np.array([0.5071, 0.4867, 0.4408]) * 255
    std = np.array([0.2675, 0.2565, 0.2761]) * 255

    normalize = transforms.NormalizeImage(mean=np.array(mean), std=np.array(std), type=np.dtype(np.float32))

    train_image_pipeline: List[Operation] = [
        RandomResizedCropRGBImageDecoder(output_size=(opt.train_size, opt.train_size), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        normalize,
        transforms.ToTensor(),
        transforms.ToTorchImage(),
    ]

    train_label_pipeline: List[Operation] = [
        IntDecoder(),
        transforms.ToTensor(),
        transforms.Squeeze(),
    ]

    train_loader = Loader(
        opt.train_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        order=OrderOption.QUASI_RANDOM,
        os_cache=False,
        drop_last=True,
        pipelines={
            'image': train_image_pipeline,
            'label': train_label_pipeline
        },
        distributed=False
    )

    val_image_pipeline: List[Operation] = [
        SimpleRGBImageDecoder(),
        normalize,
        transforms.ToTensor(),
        transforms.ToTorchImage(),
    ]

    val_label_pipeline: List[Operation] = [
        IntDecoder(),
        transforms.ToTensor(),
        transforms.Squeeze(),
    ]

    val_loader = Loader(
        opt.val_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=False,
        pipelines={
            'image': val_image_pipeline,
            'label': val_label_pipeline
        },
        distributed=False
    )

    return train_loader, val_loader

def set_model(opt):
    model = SupConResNet(name=opt.model)

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.parallel.DataParallel(model.encoder)
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)


    for p in model.parameters():
        p.requires_grad = False
    
    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        C = images.shape[-3]
        H = images.shape[-2]
        W = images.shape[-1]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            C = images.shape[-3]
            H = images.shape[-2]
            W = images.shape[-1]

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))

    if opt.save_similarity:
        if not os.path.exists('./similarities'):
            os.mkdir('similarities')
            
        similarity_folder = os.path.join('similarities', opt.dataset)
        if not os.path.exists(similarity_folder):
            os.mkdir(similarity_folder)
    
        filename = os.path.join(similarity_folder, 'similarity_matrix_topk.csv')
        sim_mat = get_similarity(train_loader, model.encoder, classifier, opt)
        
        np.savetxt(filename, sim_mat, delimiter=',', fmt='%.5f')

if __name__ == '__main__':
    main()
