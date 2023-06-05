import numpy as np
import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
import torch.nn as nn

import torchattacks
from torchattacks import PGD, PGDL2

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

import os
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from train import load_config, get_weight_dict_from_teacher, interpolate_to_student
from robust_models.wideResNet import WideResNet
from train import load_searched_model
from search import load_searched_model as load_teacher

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
config = load_config()
set_seed(config)
import time

parser = argparse.ArgumentParser()
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser.add_argument('--group_num', type=int, default=2)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--use_gpu', type=str_to_bool, default=True)
parser.add_argument('--split', type=str_to_bool, default=False)
parser.add_argument('--model_path', type=str, default='pt')
parser.add_argument('--config', type=str)

args = parser.parse_args()

def get_gradient_matrix(model, data, target, optimizer, loss_func, batch_loss=True):
    # batch_loss: Whether you calculate loss for a batch of data separately. 
    # If batch_loss=True, the loss should be in the shape of [n, 1], where loss[i] is the loss w.r.t to the i-th input.
    # You need to define a sepcific loss function for this as default loss is usually a scalar, which is the mean over all the data.
    # In this way, the forward computation is only performed once.
    # set retain_graph = True for loss.backward so that backward can be performed multiple times.
    grad_all_data = []
    if batch_loss:
        out = model(data)
        loss = loss_func(out, target, reduction='none')

    for i in range(data.shape[0]):
        #print(i, data.shape[0])
        optimizer.zero_grad()
        if batch_loss:
            loss[i].backward(retain_graph = True)
        else:
            out = model(data[i][None, :])
            #print(out.shape, target.shape)
            loss = loss_func(out[0], target[i])
            loss.backward()

        params_grad = []
        for key, value in model.named_parameters():
            if value.grad is not None:
                grad_vector = value.grad.reshape([-1])
                params_grad.append(grad_vector)

        params_grad_all = torch.cat((params_grad),0)
        params_grad_all = params_grad_all.reshape([1, -1]).cpu()
        grad_all_data.append(params_grad_all)
    
    grad_all_data_m = torch.cat((grad_all_data), 0)

    return grad_all_data_m

def get_gram_matrix_div(model, data, target, optimizer, loss_func, split_group=10):
    n = data.shape[0]
    n_group = int(n / split_group)
    gram_matrix = np.zeros([n, n])
    for i in range(split_group):
        for j in range(split_group):
            s = time.time()
            grad_i = get_gradient_matrix(model, data[i*n_group: (i+1)*n_group], target[i*n_group: (i+1)*n_group], optimizer, loss_func)
            grad_j = get_gradient_matrix(model, data[j*n_group: (j+1)*n_group], target[j*n_group: (j+1)*n_group], optimizer, loss_func)
            gram_ij = torch.matmul(grad_i, torch.transpose(grad_j, 0, 1)).numpy()
            e = time.time()
            print('ij ' ,i,  j, e-s)

    gram_matrix[i*n_group: (i+1)*n_group, j*n_group: (j+1)*n_group] = gram_ij

    return gram_matrix


if __name__ == '__main__':
    
    
    # Load your own training data
    #data = torch.load('path')
    #target = torch.load('path')

    train_loader, val_loader = create_cifar_dataloader(config, is_train=True)
    
    # cifar-s1-t1-7-adv-pgd.pt  cifar-s1-t1-7-pgd.pt  cifar-s7-0-standard-pgd.pt
    # cifar-s1-t1-3-pgd.pt cifar-s1-t1-5-pgd.pt
    # cifar-s3-0-standard-pgd.pt cifar-s5-0-standard-pgd.pt

    adv_images, adv_labels = torch.load("/scratch/un270/output/cifar-s3-t1-pgd20.pt")
    #adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    mean=torch.tensor([0.491, 0.482, 0.447])
    std=torch.tensor([0.247, 0.243, 0.262])
    adv_images = ((adv_images.float()/255)-mean[None, :, None, None])/std[None, :, None, None]
    adv_data = TensorDataset(adv_images, adv_labels)
    val_loader = DataLoader(adv_data, batch_size=10000, shuffle=False)
    
    it = iter(val_loader)
    print(len(it))
    first = next(it)
    data, target = first[0], first[1]

    print(data.shape, target.shape)
    # Load your own model
    #model = torch.load(args.model_path)
    #train_model_path = "/scratch/un270/output/cifar-small/adv/robust-s1-l1-adam-2/checkpoint_00020.pth"
    
    
    #train_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-t1-7/checkpoint_00190.pth'
    #train_model_path = '/scratch/un270/output/cifar-small-V2/train/robust-s7-0/checkpoint_00100.pth'
    #train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-s5-0-standard/checkpoint_00100.pth"
    #train_model_path = "/scratch/un270/output/cifar-small/adv/robust-s1-t1-7/checkpoint_00020.pth"

    train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-s3-0-standard/checkpoint_00100.pth" 
    #train_model_path = '/scratch/un270/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth'

    #train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-s5-0-standard/checkpoint_00100.pth" 
    #train_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-t1-5/checkpoint_00200.pth'

    #train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-l2-0-standard/checkpoint_00100.pth" 
    #train_model_path = "/scratch/un270/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
    #train_model_path = '/scratch/un270/output/cifar-small/train/robust-s1-l2-sgd-only-arch/checkpoint_00098.pth'

    path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    model_path = os.path.join(path,'/scratch/un271/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
    teacher_model = Robust_Overfitting('cuda', model_path)
    teacher_dict = get_weight_dict_from_teacher(teacher_model)
    model = create_model(config, teacher_dict)
    model = create_model(config, teacher_dict)
    checkpoint = torch.load(train_model_path, map_location=config.device)
    if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)):
       model.module.load_state_dict(checkpoint['model'])
    else:
       model.load_state_dict(checkpoint['model'])

    model = model.eval()
    '''
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.cuda()
        #mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
        #std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()
        #images = (images - mu)/std
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    ''' 

    # Define your optimizer; learning rate here does not matter
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Define your loss
    # loss_func = F.mse_loss
    loss_func = F.cross_entropy

    if args.use_gpu:
        model = model.cuda()
        data = data.cuda()
        target = target.cuda()
    
    
    # declare if you need to do the calculation in a split manner
    group_num = args.group_num
    if args.split:
        gram_matrix = get_gram_matrix_div(model, data, target, optimizer, loss_func, group_num)
    else:
        grad_all_data_m = get_gradient_matrix(model, data, target, optimizer, loss_func)
        gram_matrix = torch.matmul(grad_all_data_m, torch.transpose(grad_all_data_m, 0, 1)).numpy()
    
    gram_matrix = gram_matrix/data.shape[0]
    np.save("/scratch/un270/nkt/robust-s1-0-3-pgd20-no-mean.npy", gram_matrix)
    # todo
    #robust-s1-t1-s3-pgd20-standard.npy
    #robust-s1-t1-0-pgd20-standard-no-mean.npy
    #robust-s1-t1-0-pgd20-standard.npy

    e, v = LA.eig(gram_matrix)
    print(e.shape, e[:10])

    '''
    print("Loading npy") 
    gram_matrix1 = np.load('nkt/robust-s1-t0-7-standard.npy')
    e1, v = LA.eig(gram_matrix1)

    gram_matrix2 = np.load('nkt/robust-s1-t1-7.npy')
    e2, v2 = LA.eig(gram_matrix2)

    gram_matrix3 = np.load('nkt/robust-s1-t0-7.npy')
    e3, v3 = LA.eig(gram_matrix3)

    # show top-k eigenvalues
    k = args.top_k
    plt.plot(e1[:k],color='red', label="s1-t0-7-standard")
    plt.plot(e2[:k],color='green', label="s1-t1-7")
    plt.plot(e3[:k],color='blue', label="s1-t0-7")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig("eigen.png", dpi=300)
    '''
