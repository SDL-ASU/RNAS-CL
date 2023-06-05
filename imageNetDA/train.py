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

from fvcore.common.checkpoint import Checkpointer
import torch.nn.functional as F

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_cifar_dataloader,
    create_loss,
    create_model,
    create_optimizer,
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
from robustness import datasets
from robust_models.wideResNet import WideResNet
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
global_step = 0

from search import load_searched_model as load_search
from trades import trades_loss, multiple_adv_loss
from robustbench import load_model


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

def rkd_nas_loss(student_model, teacher_dict):
    student_model.cpu()
    s_index = 0
    GS_thetas_index = None
    loss = 0
    for k,v in student_model.state_dict().items():
        if k.find('kd_GS_thetas_index')!=-1:
           s_index = 0
           GS_thetas_index = v
        if k.find('last_conv')!=-1:
           continue
        if k.find('conv.weight')!=-1 and GS_thetas_index!=None:
           teacher_index = GS_thetas_index[s_index]           
           student_weight, teacher_weight = interpolate_to_same_shape(v, teacher_dict[int(teacher_index)])
           studeht_prob = F.softmax(student_weight)
           teacher_prob = F.softmax(teacher_weight)
           kl = (teacher_prob * torch.log(1e-6 + teacher_prob/(studeht_prob+1e-6))).mean()
           loss += kl
           s_index += 1
    student_model.cuda()
    return loss

class rkd_loss(nn.Module):
    def __init__(self, gamma=10):
        super(rkd_loss, self).__init__()
        self.gamma = gamma
        self.network_loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target, rkd_loss, teacher_output):
        #prob1 = F.softmax(teacher_output, dim=-1)
        #prob2 = F.softmax(output, dim=-1)
        #kl = ((prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)).mean()
        kl = 0.0
        ce_loss = self.network_loss_func(output, target)
        #return ce_loss
        #rkd_loss = self.gamma * rkd_nas_loss(model, self.teacher_dict)
        loss = ce_loss + kl + self.gamma * rkd_loss
        return loss, ce_loss, rkd_loss

class trades_rkd_loss(nn.Module):
    def __init__(self, gamma=10):
        super(trades_rkd_loss, self).__init__()
        self.gamma = gamma

    def forward(self, data, output, target, rkd_loss, teacher_output, model, optimizer):
        ce_loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=0.0078,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=6,
                           distance='l_inf')
        prob1 = F.softmax(teacher_output, dim=-1)
        prob2 = F.softmax(output, dim=-1)
        kl = ((prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)).mean()
        loss = ce_loss + kl + self.gamma * rkd_loss
        return loss, ce_loss, rkd_loss

class multi_adv_loss(nn.Module):
    def __init__(self, gamma=10):
        super(multi_adv_loss, self).__init__()
        self.gamma = gamma

    def forward(self, data, output, target, rkd_loss, teacher_output, model, optimizer, logger):
        ce_loss = multiple_adv_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=0.0078,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=6,
                           distance='l_inf', logger=logger)

        prob1 = F.softmax(teacher_output, dim=-1)
        prob2 = F.softmax(output, dim=-1)
        kl = ((prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)).mean()
        loss = ce_loss + kl + self.gamma * rkd_loss
        return loss, ce_loss, rkd_loss

def get_weight_dict_from_teacher(teacher_model):
    teacher_dict = {}
    index = 0
    for k,v in teacher_model.state_dict().items():
        if k.find('conv')!=-1:
           teacher_dict[index] = v
           index+=1
    return teacher_dict


def load_searched_model(model, path, device):
    state_dict = torch.load(path, map_location=device)['model']
    for k,v in state_dict.items():
        if(k.find("kd_GS_thetas")!=-1):
           model.state_dict()[k] = v
           '''
           gumbel_index = torch.ones(1, 1)
           soft_mask_variables = nn.functional.gumbel_softmax(v, 0.05, hard=True)
           index = torch.argmax(soft_mask_variables)
           gumbel_index[0][0] = index
           i = k.find("kd_GS_thetas")
           key = k[:i] + "kd_GS_thetas_index" + k[i+12:]
           model.state_dict()[key].copy_(gumbel_index)
           '''
        elif(k.find("GS_thetas")!=-1):
           #i = k.find("GS_thetas")
           #key = k[:i] + "GS_thetas_index"
           #model.state_dict()[key].copy_(v)

           soft_mask_variables = nn.functional.gumbel_softmax(v, 0.055, hard=True)
           index = torch.argmax(soft_mask_variables)
           i = k.find("GS_thetas")
           key = k[:i] + "GS_thetas_index" + k[i+9:]
           model.state_dict()[key].copy_(index)
        else:
            model.state_dict()[k] = v
    return model


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gt', type=float)
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
    #config.train.gamma_train = args.gt
    #config.train.output_dir = "/scratch/un270/output/imagenet-v8/train/r18-1e-s1-t-"+str(args.gt)
    config.freeze()
    return config


def subdivide_batch(config, data, targets):

    subdivision = config.train.subdivision

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in zip(targets1.chunk(subdivision), targets2.chunk(subdivision))]
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


def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader,
          logger, tensorboard_writer, tensorboard_writer2, teacher_model):
    global global_step

    logger.info(f'Train {epoch} {global_step}')

    device = torch.device(config.device)

    model.train()

    mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda(config.device)
    std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda(config.device)

    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    rkd_loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(train_loader):
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

        data_chunks, target_chunks = subdivide_batch(config, data, targets)

        optimizer.zero_grad()
        outputs = []
        losses = []


        for data_chunk, target_chunk in zip(data_chunks, target_chunks):
            with torch.no_grad():
                 all_output, teacher_output = teacher_model(data_chunk)
                 #teacher_output = teacher_output[:,[15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]]

            if config.augmentation.use_dual_cutout:
                w = data_chunk.size(3) // 2
                data1 = data_chunk[:, :, :, :w]
                data2 = data_chunk[:, :, :, w:]
                outputs1 = model(data1)
                outputs2 = model(data2)
                output_chunk = torch.cat(
                    (outputs1.unsqueeze(1), outputs2.unsqueeze(1)), dim=1)
            else:
                #data_chunk_norm = (data_chunk - mu)/std
                output_chunk, _, kl_chunk = model(data_chunk, all_output, 0.055)
            outputs.append(output_chunk)
            
            loss, ce_loss, rkd_loss = loss_func(data_chunk, output_chunk, target_chunk, kl_chunk, teacher_output, model, optimizer, logger)
            #loss, ce_loss, rkd_loss = loss_func(output_chunk, target_chunk, kl_chunk, teacher_output)
            loss, ce_loss, rkd_loss = loss.mean(), ce_loss.mean(), rkd_loss.mean()
            losses.append(loss)
            loss.backward()
            '''
            if config.device != 'cpu':
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            '''
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
        optimizer.step()

        acc1, acc5 = compute_accuracy(config,
                                      outputs,
                                      targets,
                                      augmentation=True,
                                      topk=(1, 5))

        loss = sum(losses)

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
        ce_loss = ce_loss.item()
        rkd_loss = rkd_loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()

        num = data.size(0)

        loss_meter.update(loss, num)
        ce_loss_meter.update(ce_loss, num)
        rkd_loss_meter.update(rkd_loss, num)
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
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    f'ce_loss {ce_loss_meter.val:.4f} ({ce_loss_meter.avg:.4f}) '
                    f'kl {rkd_loss_meter.val:.8f} ({rkd_loss_meter.avg:.8f}) '
                    f'rkd_loss {config.train.gamma_train * rkd_loss_meter.val:.4f} ({config.train.gamma_train * rkd_loss_meter.avg:.4f}) ' 
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
             tensorboard_writer, teacher_model):
    logger.info(f'Val {epoch}')

    device = torch.device(config.device)

    model.eval()

    mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda(config.device)
    std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda(config.device)

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    rkd_loss_meter = AverageMeter()
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

            data = data.to(
                device, non_blocking=config.validation.dataloader.non_blocking)
            targets = targets.to(device)
           
            #data_norm = (data - mu)/std
            all_output, teacher_output = teacher_model(data)
            #teacher_output = teacher_output[:,[15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]]

            outputs, _, kl_chunks = model(data, all_output, 0.055)
            loss, ce_loss, rkd_loss = loss_func(outputs, targets, kl_chunks, teacher_output)
            loss, ce_loss, rkd_loss = loss.mean(), ce_loss.mean(), rkd_loss.mean()

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
            rkd_loss = rkd_loss.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)
            rkd_loss_meter.update(rkd_loss, num)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        logger.info(f'Epoch {epoch} '
                    f'loss {loss_meter.avg:.4f} '
                    f'kl_loss {rkd_loss_meter.avg:.4f} '
                    f'rkd_loss {config.train.gamma_train * rkd_loss_meter.avg:.4f} '
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
    global global_step, train_type
    train_type = "train"

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
        '''
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            #save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            #if diff is not None:
            #    save_config(diff, output_dir / 'config_min.yaml')
         '''
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

    #model = create_model(config)
    #model = load_searched_model(model, '/home/utkarsh/output/search_standard_cifar_10/checkpoint_00100.pth')
    #macs, n_params = count_op(config, model)
    #logger.info(f'MACs   : {macs}')
    #logger.info(f'#params: {n_params}')

    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
   
    #teacher_model = WideResNet()
    #teacher_model.load_state_dict(torch.load("robust_models/model_cifar_wrn.pt",map_location=config.device))
    
    #model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/cifar_model_weights_30_epochs.pth')
    #teacher_model = Fast_AT('cuda', model_path)
    
    model_path = os.path.join(path,'/scratch/ag7644/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
    teacher_model = Robust_Overfitting('cuda', model_path)

    teacher_model.load()
    #teacher_model = teacher_model.cuda(config.device)
    #################################################################
    #teacher_model = load_model(model_name='Engstrom2019Robustness', dataset='cifar10', threat_model='Linf') 

    #path_wide_Resnet_2 = "/scratch/un270/wide_resnet50_2_l2_eps1.ckpt"
    #dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
    #model = dataset.get_model('wide_resnet50_2', False)
    #_, _, teacher_model = load_search(model, path_wide_Resnet_2)

    #path_Resnet_18 = "/scratch/un270/resnet18_l2_eps1.ckpt"
    #dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
    #model = dataset.get_model('resnet18', False)
    #_, _, teacher_model = load_search(model, path_Resnet_18)

    #path_Resnet_50 = "/scratch/un270/resnet50_l2_eps1.ckpt"
    #dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
    #model = dataset.get_model('resnet50', False)
    #_, _, teacher_model = load_search(model, path_Resnet_50)

    teacher_dict = get_weight_dict_from_teacher(teacher_model)
    model = create_model(config, teacher_dict)
    '''
    model = load_searched_model(model, '/scratch/un270/output/cifar10-V3/search-with-kl/fast_1e_6/checkpoint_00200.pth', config.device)
    # Calculate number of conv layers
    teacher_dict = {}
    index=0
    for k,v in model.state_dict().items():
        if k.find("conv.weight")!=-1:
           curr = None
           for kt, kv in teacher_model.state_dict().items():
               if kt.find("conv")!=-1:
                  student_weight, teacher_weight = interpolate_to_student(v, kv)
                  if curr == None:
                     curr = teacher_weight[None, :]
                  else:
                     curr = torch.cat((curr, teacher_weight[None, :]), 0)
           teacher_dict[index] = curr
           index+=1

    for key, value in teacher_dict.items():
        teacher_dict[key] = teacher_dict[key].cuda(config.device)
    ''' 
    model = create_model(config, teacher_dict)
    #macs, n_params = count_op(config, model)
    #logger.info(f'MACs   : {macs}')
    #logger.info(f'#params: {n_params}')
   
    #path = '/scratch/ag7644/output/cifar-small/search/robust-s1-7/checkpoint_00100.pth'
    #model = load_searched_model(model, path, config.device)
    
    #model = load_searched_model(model, '/scratch/ag7644/output/cifar-small-V4/search/robust-s7-standard/checkpoint_00100.pth', config.device) 
    #model = load_searched_model(model,'/scratch/un270/output/cifar-small-V3/search/robust-s7-s1/checkpoint_00100.pth', config.device) 
    

    #train_model_path = '/scratch/un270/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth'
    #train_model_path = "/scratch/un270/output/imagenet-v8/train/r18-1e-s1-t-1.0/checkpoint_00200.pth" 
    #train_model_path = "/scratch/un270/output/imagenet-v8/train/wrt-50-2-e1-s1-t1e-1/checkpoint_00200.pth"
    #train_model_path = "/scratch/un270/output/imagenet-v8/train/r50-s1-t-1.0/checkpoint_00200.pth"
   
    
    #train_model_path = '/scratch/ag7644/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth' 
    train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-5/checkpoint_00200.pth'
    #train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-7/checkpoint_00190.pth' 
    #train_model_path = '/scratch/ag7644/output/cifar-small/train/robust-s1-l2-sgd/checkpoint_00100.pth'
    #train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
    #train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l1-sgd/checkpoint_00100.pth"

    #train_model_path = "/scratch/ag7644/output/cifar-small-V7/adv/robust-wrt-l2-pgd-consit-1e-1/checkpoint_00020.pth" 
    checkpoint = torch.load(train_model_path, map_location=config.device)
    if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)):
       model.module.load_state_dict(checkpoint['model'])
    else:
       model.load_state_dict(checkpoint['model'])
    
    
    optimizer = create_optimizer(config, model)

    #if config.device != 'cpu':
    #    model.to('cuda')
    #     model, optimizer = apex.amp.initialize(model, optimizer, opt_level=config.train.precision)
    # model = apply_data_parallel_wrapper(config, model)

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

    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
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

    #train_loss, val_loss = create_loss(config)
    #train_loss = rkd_loss(config.train.gamma_train)
    #train_loss = trades_rkd_loss(config.train.gamma_train)
    train_loss = multi_adv_loss(config.train.gamma_train)
    val_loss = rkd_loss(config.train.gamma_train)


    '''
    if (config.train.val_period > 0 and start_epoch == 0
            and config.train.val_first):
        validate(0, config, model, val_loss, val_loader, logger,
                 tensorboard_writer, teacher_model)
   '''
    #validate(0, config, model, val_loss, val_loader, logger,tensorboard_writer, teacher_model)
    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1

        np.random.seed(seed)
        train(epoch, config, model, optimizer, scheduler, train_loss,
              train_loader, logger, tensorboard_writer, tensorboard_writer2, teacher_model)

        if config.train.val_period > 0 and (epoch %
                                            config.train.val_period == 0):
            validate(epoch, config, model, val_loss, val_loader, logger,
                     tensorboard_writer, teacher_model)

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
#train_type = None

if __name__ == '__main__':
    main()
