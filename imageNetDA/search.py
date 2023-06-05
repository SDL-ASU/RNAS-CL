#!/usr/bin/env python

import argparse
import pathlib
import time

#import apex
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_cifar_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_optimizer_search,
    create_scheduler,
    get_default_config,
    update_config,
)

from pytorch_image_classification.config.config_node import ConfigNode

from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)

from robust_models.wideResNet import WideResNet  
from robustness import datasets
global_step = 0

import os
from robust_cifar10_models.at_he import AT_HE
from robust_cifar10_models.awp import AWP
from robust_cifar10_models.fast_at import Fast_AT
from robust_cifar10_models.featurescatter import Feature_Scatter
from robust_cifar10_models.hydra import Hydra
from robust_cifar10_models.label_smoothing import Label_Smoothing
from robust_cifar10_models.pre_training import Pre_Training
from robust_cifar10_models.robust_overfiting import Robust_Overfitting
from robust_cifar10_models.rst import RST
from robust_cifar10_models.trades import WideResNet_TRADES

from robustbench import load_model

student_shape = [ 16, 16, 16,],[ 16, 16, 16,],[ 16, 16, 16,],[ 96, 16, 16,],[ 96, 8, 8,],[ 28, 8, 8,],[ 168, 8, 8,],[ 168, 8, 8,],[ 28, 8, 8,],[ 168, 8, 8,],[ 168, 8, 8,],[ 28, 8, 8,],[ 168, 8, 8,],[ 168, 4, 4,],[ 40, 4, 4,],[ 240, 4, 4,],[ 240, 4, 4,],[ 40, 4, 4,],[ 240, 4, 4,],[ 240, 4, 4,],[ 40, 4, 4,],[ 240, 4, 4,],[ 240, 2, 2,],[ 96, 2, 2,],[ 576, 2, 2,],[ 576, 2, 2,],[ 96, 2, 2,],[ 576, 2, 2,],[ 576, 2, 2,],[ 96, 2, 2,],[ 576, 2, 2,],[ 576, 2, 2,],[ 128, 2, 2,],[ 768, 2, 2,],[ 768, 2, 2,],[ 128, 2, 2,],[ 768, 2, 2,],[ 768, 2, 2,],[ 128, 2, 2,],[ 768, 2, 2,],[ 768, 2, 2,],[ 128, 2, 2,],[ 768, 2, 2,],[ 768, 1, 1,],[ 216, 1, 1,],[ 1296, 1, 1,],[ 1296, 1, 1,],[ 216, 1, 1,],[ 1296, 1, 1,],[ 1296, 1, 1,],[ 216, 1, 1,],[ 1296, 1, 1,],[ 1296, 1, 1,],[ 216, 1, 1,],[ 1296, 1, 1,],[ 1984, 1, 1,],


def interpolate_to_same_shape(student_weight, teacher_weight):
       min_kernel = student_weight.shape[3]
       if student_weight.shape[3]>teacher_weight.shape[3]:
          min_kernel = teacher_weight.shape[3]
          student_weight = F.interpolate(student_weight, size=([teacher_weight.shape[3],teacher_weight.shape[3]]),mode='bilinear')
       elif student_weight.shape[3]<teacher_weight.shape[3]:
          teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[3],student_weight.shape[3]]),mode='bilinear')
       teacher_weight = teacher_weight.reshape(teacher_weight.shape[0], teacher_weight.shape[1], -1)
       student_weight = student_weight.reshape(student_weight.shape[0], student_weight.shape[1], -1)
       teacher_weight = teacher_weight.permute(0,2,1)
       student_weight = student_weight.permute(0,2,1)

       if student_weight.shape[2]>teacher_weight.shape[2]:
          student_weight = F.interpolate(student_weight,size=([teacher_weight.shape[2]]), mode='linear')
       elif student_weight.shape[2]<teacher_weight.shape[2]:
          teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[2]]), mode='linear')
       teacher_weight = teacher_weight.permute(1,2,0)
       student_weight = student_weight.permute(1,2,0)

       if student_weight.shape[2]>teacher_weight.shape[2]:
          student_weight = F.interpolate(student_weight,size=([teacher_weight.shape[2]]), mode='linear')
       elif student_weight.shape[2]<teacher_weight.shape[2]:
          teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[2]]), mode='linear')
       teacher_weight = teacher_weight.permute(2,1,0)
       student_weight = student_weight.permute(2,1,0)

       teacher_weight = teacher_weight.reshape(teacher_weight.shape[0], teacher_weight.shape[1], min_kernel, min_kernel)
       student_weight = student_weight.reshape(student_weight.shape[0], student_weight.shape[1], min_kernel, min_kernel)

       return student_weight, teacher_weight

def interpolate_to_student(student_weight, teacher_weight):
       min_kernel = student_weight.shape[3]
       teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[3],student_weight.shape[3]]),mode='bilinear')
       teacher_weight = teacher_weight.reshape(teacher_weight.shape[0], teacher_weight.shape[1], -1)
       student_weight = student_weight.reshape(student_weight.shape[0], student_weight.shape[1], -1)
       teacher_weight = teacher_weight.permute(0,2,1)
       student_weight = student_weight.permute(0,2,1)

       teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[2]]), mode='linear')
       teacher_weight = teacher_weight.permute(1,2,0)
       student_weight = student_weight.permute(1,2,0)

       teacher_weight = F.interpolate(teacher_weight, size=([student_weight.shape[2]]), mode='linear')
       teacher_weight = teacher_weight.permute(2,1,0)
       student_weight = student_weight.permute(2,1,0)

       teacher_weight = teacher_weight.reshape(teacher_weight.shape[0], teacher_weight.shape[1], min_kernel, min_kernel)
       student_weight = student_weight.reshape(student_weight.shape[0], student_weight.shape[1], min_kernel, min_kernel)

       return student_weight, teacher_weight

def interpolate_to_student_output(c, h_s, w_s, teacher_output):
       teacher_height = teacher_output.shape[3]
       teacher_output = teacher_output.reshape(teacher_output.shape[0], teacher_output.shape[1], -1)
       teacher_output = teacher_output.permute(0,2,1)
       teacher_output = F.interpolate(teacher_output, size=([c]), mode='linear')
       teacher_output = teacher_output.permute(0,2,1)
       teacher_output = teacher_output.reshape(teacher_output.shape[0], teacher_output.shape[1], teacher_height, teacher_height)
       teacher_output = F.interpolate(teacher_output, size=([h_s,w_s]),mode='bilinear')
       return teacher_output

def rkd_nas_loss(student_model, teacher_model, temperature = 0.5):
    teacher_model.cpu()
    student_model.cpu()
    s_index = 0
    soft_mask_variables = None
    loss = 0
    for k,v in student_model.state_dict().items():
        if k.find('kd_GS_thetas')!=-1:
           soft_mask_variables = nn.functional.gumbel_softmax(v, temperature)
           s_index = 0
        if k.find('last_conv')!=-1:
           continue
        if k.find('conv.weight')!=-1 and soft_mask_variables!=None:
           t_index = 0
           for kt, vt in teacher_model.state_dict().items():
               if kt.find('conv')!=-1:
                  student_weight, teacher_weight = interpolate_to_same_shape(v, vt)
                  studeht_prob = F.softmax(student_weight)
                  teacher_prob = F.softmax(teacher_weight)
                  kl = (teacher_prob * torch.log(1e-10 + teacher_prob/(studeht_prob+1e-10))).mean()
                  loss += kl * soft_mask_variables[s_index][t_index]
                  t_index+=1
           s_index += 1
    student_model.cuda()
    return loss

def get_weight_dict_from_teacher(teacher_model):
    teacher_dict = {}
    index = 0
    for k,v in teacher_model.state_dict().items():
        if k.find('conv')!=-1:
           teacher_dict[index] = v
           index+=1
    return teacher_dict

class search_loss(nn.Module):
    def __init__(self, alpha, beta, network_loss_func, gamma):
        super(search_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.network_loss_func = network_loss_func

    def forward(self, output, teacher_output, cost, target, rkd_loss):
        #rkd_loss.requires_grad=True
        #prob1 = F.softmax(teacher_output, dim=-1)
        #prob2 = F.softmax(output, dim=-1)
        #kl = ((prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)).mean()
        kl = 0.0
        if rkd_loss!=None:
           ce_loss = self.network_loss_func(output, target)
           #network_loss = ce_loss + self.gamma*rkd_loss
           network_loss = ce_loss + self.gamma*rkd_loss + kl 
        else:
           network_loss = self.network_loss_func(output, target)
        latency_loss = torch.log(torch.max(cost) ** self.beta)
        loss = self.alpha * network_loss * latency_loss
        if rkd_loss!=None:
           return loss, ce_loss, rkd_loss, kl, latency_loss
        else:
           return loss, network_loss, latency_loss

def load_searched_model(model, path):
    state_dict = torch.load(path)['model']
    for k,v in state_dict.items():
        key = k[7:]
        if key=='normalizer.new_mean':
           mean = v
        if key=='normalizer.new_std':
           std = v
        if key.split('.')[0]=='model':
           model.state_dict()[key[6:]].copy_(v)
    return mean, std, model

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gs', type=float)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    #config.train.gamma_search = args.gs
    #config.train.output_dir = "/scratch/un270/output/imagenet-v8/search/r18-1e+"+str(args.gs) 
    config.freeze()
    return config


def subdivide_batch(config, data, targets):
    subdivision = config.train.subdivision

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in
                         zip(targets1.chunk(subdivision), targets2.chunk(subdivision))]
    elif config.augmentation.use_ricap:
        target_list, weights = targets
        target_list_chunks = list(
            zip(*[target.chunk(subdivision) for target in target_list]))
        target_chunks = [(chunk, weights) for chunk in target_list_chunks]
    else:
        target_chunks = targets.chunk(subdivision)
    return data_chunks, target_chunks


def send_targets_to_device(config, targets, device):
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        t1, t2, lam = targets
        targets = (t1.to(device), t2.to(device), lam)
    elif config.augmentation.use_ricap:
        labels, weights = targets
        labels = [label.to(device) for label in labels]
        targets = (labels, weights)
    else:
        targets = targets.to(device)
    return targets


def train(epoch, config, model, optimizer, gs_optimizer, scheduler, gs_scheduler, temperature, loss_func, train_loader,
          logger, tensorboard_writer, tensorboard_writer2, teacher_model):
    global global_step

    logger.info(f'Train {epoch} {global_step}')

    device = torch.device(config.device)

    model.train()

    mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda(config.device)
    std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda(config.device)

    loss_meter = AverageMeter()

    network_loss_meter = AverageMeter()
    rkd_loss_meter = AverageMeter()
    cost_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()

    # iteration in one epoch
    for step, (data, targets) in enumerate(train_loader):
        if step < len(train_loader) * (1 - config.gs_search.search_data_percent):
            train_network = True
            search_network = False
        else:
            train_network = False
            search_network = True

        step += 1
        global_step += 1

        if get_rank() == 0 and step == 1:
            if config.tensorboard.train_images:
                image = torchvision.utils.make_grid(data,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Train/Image', image, epoch)

        data = data.to(device, non_blocking=config.train.dataloader.non_blocking)
        targets = send_targets_to_device(config, targets, device)
        #for key in all_output.keys():
        #    all_output[key] = all_output[key].to(device)

        data_chunks, target_chunks = subdivide_batch(config, data, targets)

        optimizer.zero_grad()
        gs_optimizer.zero_grad()

        outputs = []
        costs = []
        losses = []
        network_losses = []
        rkd_losses = []
        kl_losses = []

        s4 = time.time()
        for data_chunk, target_chunk in zip(data_chunks, target_chunks):
            start = time.time()
            #print("start",s4-start)
            s = time.time()
            with torch.no_grad():
                 all_output, teacher_output = teacher_model(data_chunk)
                 #teacher_output = teacher_output[:,[15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]]
            s1 = time.time()
            
            #data_chunk_norm = (data_chunk - mu)/std
            output_chunk, cost_chunk, kl_chunk = model(data_chunk, all_output, temperature)
            s2 = time.time()
            #print("model", s2-s1)
            end = time.time()
            #print("qqq ",end-start)
            
            outputs.append(output_chunk)
            #rkd_nas_loss1 = rkd_nas_loss(model, teacher_model, temperature)
            
            end1 = time.time()
            
            #rkd_nas_loss1 = None
            
            #print("qwqe ",end1-end)
            search_loss_func = search_loss(config.gs_search.alpha, config.gs_search.beta, loss_func, config.train.gamma_search)

            loss, network_loss, rkd_loss, kl_loss, latency_loss = search_loss_func(output_chunk, teacher_output, cost_chunk, target_chunk, kl_chunk)
            loss, network_loss, rkd_loss, kl_loss, latency_loss = loss.mean(), network_loss.mean(), rkd_loss.mean(), kl_loss, latency_loss.mean() 
            losses.append(loss)
            network_losses.append(network_loss)
            rkd_losses.append(rkd_loss)
            costs.append(latency_loss)
            kl_losses.append(kl_loss)

            loss.backward()
            s3 = time.time()
            #print("loss", s3-s2)

        outputs = torch.cat(outputs)

        if config.train.gradient_clip > 0:
            if config.device != 'cpu':
                pass
                #torch.nn.utils.clip_grad_norm_(
                #    apex.amp.master_params(optimizer),
                #    config.train.gradient_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.train.gradient_clip)
        if config.train.subdivision > 1:
            for param in model.parameters():
                param.grad.data.div_(config.train.subdivision)

        if train_network:
            optimizer.step()

        if search_network:
            gs_optimizer.step()

        acc1, acc5 = compute_accuracy(config,
                                      outputs,
                                      targets,
                                      augmentation=True,
                                      topk=(1, 5))

        loss = sum(losses)
        rkd_loss = sum(rkd_losses)
        cost = sum(costs)
        network_loss = sum(network_losses)
        kl_loss = sum(kl_losses)

        if config.train.distributed:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            cost_all_reduce = dist.all_reduce(cost,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            network_loss_all_reduce = dist.all_reduce(network_loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc1_all_reduce = dist.all_reduce(acc1,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc5_all_reduce = dist.all_reduce(acc5,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            loss_all_reduce.wait()
            cost_all_reduce.wait()
            network_loss_all_reduce.wait()
            acc1_all_reduce.wait()
            acc5_all_reduce.wait()
            loss.div_(dist.get_world_size())
            cost.div_(dist.get_world_size())
            network_loss.div_(dist.get_world_size())
            acc1.div_(dist.get_world_size())
            acc5.div_(dist.get_world_size())

        loss = loss.item()
        rkd_loss = rkd_loss.item()
        #kl_loss = kl_loss.item()
        cost = cost.item()
        network_loss = network_loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()

        num = data.size(0)

        loss_meter.update(loss, num)
        rkd_loss_meter.update(rkd_loss, num)
        kl_loss_meter.update(kl_loss, num)
        cost_meter.update(cost, num)
        network_loss_meter.update(network_loss, num)
        acc1_meter.update(acc1, num)
        acc5_meter.update(acc5, num)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if get_rank() == 0:
            if step % config.train.log_period == 0 or step == len(
                    train_loader):
                logger.info(
                    f'Epoch {epoch} '
                    f'Step {step}/{len(train_loader)} '
                    f'lr {scheduler.get_last_lr()[0]:.6f} '
                    f'loss {loss_meter.val:.8f} ({loss_meter.avg:.8f}) '
                    f'cost {cost_meter.val:.4f} ({cost_meter.avg:.4f}) '
                    f'network_loss {network_loss_meter.val:.4f} ({network_loss_meter.avg:.4f}) '
                    f'intermediate-kl {rkd_loss_meter.val:.8f} ({rkd_loss_meter.avg:.8f}) '
                    f'rkd_loss {config.train.gamma_search*rkd_loss_meter.val:.4f} ({config.train.gamma_search*rkd_loss_meter.avg:.4f}) '
                    f'kl {kl_loss_meter.val:.4f} ({kl_loss_meter.avg:.4f}) '
                    f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                    f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

                tensorboard_writer2.add_scalar('Train/RunningLoss',
                                               loss_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc1',
                                               acc1_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc5',
                                               acc5_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningLearningRate',
                                               scheduler.get_last_lr()[0],
                                               global_step)

        scheduler.step()
        gs_scheduler.step()
        s4 = time.time()

    if get_rank() == 0:
        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

        tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc5', acc5_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
        tensorboard_writer.add_scalar('Train/LearningRate',
                                      scheduler.get_last_lr()[0], epoch)


def validate(epoch, config, model, loss_func, val_loader, logger,
             tensorboard_writer, temperature, teacher_model):
    logger.info(f'Val {epoch}')

    device = torch.device(config.device)

    model.eval()

    mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda(config.device)
    std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda(config.device)

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(val_loader):
            if get_rank() == 0:
                if config.tensorboard.val_images:
                    if epoch == 0 and step == 0:
                        image = torchvision.utils.make_grid(data,
                                                            normalize=True,
                                                            scale_each=True)
                        tensorboard_writer.add_image('Val/Image', image, epoch)

            data = data.to(device, non_blocking=config.validation.dataloader.non_blocking)
            targets = targets.to(device)
            with torch.no_grad():
                 all_output, teacher_output = teacher_model(data)
                 #teacher_output = teacher_output[:,[15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]]

    
            #data_norm = (data - mu)/std
            outputs, cost, kl_chunk = model(data, all_output, temperature)
            loss = loss_func(outputs, targets)

            acc1, acc5 = compute_accuracy(config,
                                          outputs,
                                          targets,
                                          augmentation=False,
                                          topk=(1, 5))

            if config.train.distributed:
                loss_all_reduce = dist.all_reduce(loss,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                acc1_all_reduce = dist.all_reduce(acc1,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                acc5_all_reduce = dist.all_reduce(acc5,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                loss_all_reduce.wait()
                acc1_all_reduce.wait()
                acc5_all_reduce.wait()
                loss.div_(dist.get_world_size())
                acc1.div_(dist.get_world_size())
                acc5.div_(dist.get_world_size())

            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        logger.info(f'Epoch {epoch} '
                    f'network_loss {loss_meter.avg:.4f} '
                    f'acc@1 {acc1_meter.avg:.4f} '
                    f'acc@5 {acc5_meter.avg:.4f}')

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

    if get_rank() == 0:
        if epoch > 0:
            tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc1', acc1_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc5', acc5_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
        if config.tensorboard.model_params:
            for name, param in model.named_parameters():
                tensorboard_writer.add_histogram(name, param, epoch)



def main():

    global global_step
   
    config = load_config()
    
    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)

    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)

    output_dir = pathlib.Path(config.train.output_dir)
    
    if get_rank() == 0:
        if not config.train.resume and output_dir.exists():
            raise RuntimeError(
                f'Output directory `{output_dir.as_posix()}` already exists')
        output_dir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            #save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            if diff is not None:
                save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    #logger.info(get_env_info(config))

# Load Data
    if config.dataset.name == 'CIFAR10':
       train_loader, val_loader = create_cifar_dataloader(config, is_train=True)
    else:
       train_loader, val_loader = create_dataloader(config, is_train=True)

# Create Network Model
    #model = create_model(config)
    # macs, n_params = count_op(config, model)
    # logger.info(f'MACs   : {macs}')
    # logger.info(f'#params: {n_params}')

    temperature = config.gs_search.base_temperature
    
    torch.cuda.empty_cache()
    #teacher_model = models.resnet18(pretrained=True)
    
    ################################ CIFAR Models ###################
    #teacher_model = WideResNet()
    #teacher_model = teacher_model.cuda(config.device)
    #teacher_model.load_state_dict(torch.load("robust_models/model_cifar_wrn.pt"))
    
    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    #model_path = os.path.join(path,'/home/unath/RKD_NAS/DAS/model/ciafar10_checkpoints/model-wideres-pgdHE-wide10.pt')
    #teacher_model = AT_HE(config.device, model_path) 
    #teacher_model.load()

    #model_path = os.path.join(path,'/home/unath/RKD_NAS/DAS/model/ciafar10_checkpoints/RST-AWP_cifar10_linf_wrn28-10.pt')
    #teacher_model = AWP('cuda', model_path)
    #teacher_model.load()

    #model_path = '/scratch/un270/ciafar10_robust_checkpoints/checkpoint-199-ipot'
    #teacher_model = Feature_Scatter('cuda', model_path)

    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/cifar_model_weights_30_epochs.pth')
    #teacher_model = Fast_AT('cuda', model_path)
    
    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/model_best_dense.pth.tar')
    #teacher_model = Hydra('cuda', model_path)

    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/model_best.pth')
    #teacher_model = Label_Smoothing('cuda', model_path)

    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/cifar10wrn_baseline_epoch_4.pt')
    #teacher_model = Pre_Training('cuda', model_path)

    model_path = os.path.join(path,'/scratch/ag7644/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
    teacher_model = Robust_Overfitting('cuda', model_path)

    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/cifar10_rst_adv.pt.ckpt')
    #teacher_model = RST('cuda', model_path)

    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/model_cifar_wrn.pt')
    #teacher_model = WideResNet_TRADES('cuda', model_path)
 
    teacher_model.load()
    
    #################################################################
    
    #teacher_model = load_model(model_name='Engstrom2019Robustness', dataset='cifar10', threat_model='Linf')

    #path_wide_Resnet_2 = "/scratch/un270/wide_resnet50_2_l2_eps1.ckpt"
    #dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
    #model = dataset.get_model('wide_resnet50_2', False)
    #_, _, teacher_model = load_searched_model(model, path_wide_Resnet_2)

    #path_Resnet_18 = "/scratch/un270/resnet18_l2_eps1.ckpt"
    #dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
    #model = dataset.get_model('resnet18', False)
    #_, _, teacher_model = load_searched_model(model, path_Resnet_18)

    teacher_dict = get_weight_dict_from_teacher(teacher_model)

    model = create_model(config, teacher_dict)
    macs, n_params = count_op(config, model)
    logger.info(f'MACs   : {macs}')
    logger.info(f'#params: {n_params}')
    
    model = create_model(config, teacher_dict)   
    

# Create Optimizer
    optimizer, gs_optimizer = create_optimizer_search(config, model)

    #if config.device != 'cpu':
    #    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=config.train.precision)
    #    model, gs_optimizer = apex.amp.initialize(model, gs_optimizer, opt_level=config.train.precision)

    if config.train.distributed:
        model = apply_data_parallel_wrapper(config, model)
        teacher_model = teacher_model.cuda()
        #teacher_model = apply_data_parallel_wrapper(config, teacher_model)

    ngpu=1
    if ngpu>1:
       model = nn.DataParallel(model)
       teacher_model = nn.DataParallel(teacher_model)
       #for key, value in teacher_dict.items():
       #    teacher_dict[key] = nn.DataParallel(teacher_dict[key])
    else:
       model = model.cuda(config.device)
    #teacher_dict = get_weight_dict_from_teacher(teacher_model)
    teacher_model = teacher_model.cuda(config.device)
    model = model.cuda(config.device)
    teacher_model.eval()

# Create LR Scheduler
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    gs_scheduler = create_scheduler(config, gs_optimizer, steps_per_epoch=len(train_loader))

# Create checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
# Resume training if needed
    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()
    elif config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
            config, output_dir, purge_step=config.train.start_epoch + 1)
        tensorboard_writer2 = create_tensorboard_writer(
            config, output_dir / 'running', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()

# Create Loss Function

    train_loss, val_loss = create_loss(config)
#    search_model_path = '/scratch/un270/output/cifar10-V3/search-with-kl/fast_1e_6/checkpoint_00200.pth'
#    checkpoint = torch.load(search_model_path, map_location=config.device)
#    if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)):
#       model.module.load_state_dict(checkpoint['model'])
#    else:
# model.load_state_dict(checkpoint['model'])

# Start Training
#     if (config.train.val_period > 0 and start_epoch == 0
#             and config.train.val_first):
#    validate(200, config, model, val_loss, val_loader, logger,tensorboard_writer, 0.0006170490204334047)
    validate(100, config, model, val_loss, val_loader, logger,
                     tensorboard_writer, temperature, teacher_model)
    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1

        np.random.seed(seed)
        train(epoch, config, model, optimizer, gs_optimizer, scheduler, gs_scheduler, temperature, train_loss,
              train_loader, logger, tensorboard_writer, tensorboard_writer2, teacher_model)

        if config.train.val_period > 0 and (epoch % config.train.val_period == 0):
            validate(epoch, config, model, val_loss, val_loader, logger,
                     tensorboard_writer, temperature, teacher_model)

        temperature = temperature * np.exp(config.gs_search.temp_factor)
        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        if (epoch % config.train.checkpoint_period == 0) or (
                epoch == config.scheduler.epochs):
            checkpoint_config = {
                'epoch': epoch,
                'global_step': global_step,
                'config': config.as_dict(),
            }
            checkpointer.save(f'checkpoint_{epoch:05d}', **checkpoint_config)

    tensorboard_writer.close()
    tensorboard_writer2.close()

#global train_type
#train_type = "search"

if __name__ == '__main__': 
   main()
