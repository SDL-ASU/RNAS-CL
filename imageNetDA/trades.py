import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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

import pathlib
from myattack import cw_whitebox , pgd_whitebox, fgsm_attack

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv, None, 0.055), dim=1),
                                       F.softmax(model(x_natural, None, 0.055), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def multiple_adv_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf', logger=None, tem=0.1, conf=0.2):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial examplei

    x_adv1 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #x_adv2 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv1.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv1, None, 0.055), dim=1),
                                       F.softmax(model(x_natural, None, 0.055), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv1])[0]
        x_adv1 = x_adv1.detach() + step_size * torch.sign(grad.detach())
        x_adv1 = torch.min(torch.max(x_adv1, x_natural - epsilon), x_natural + epsilon)
        x_adv1 = torch.clamp(x_adv1, 0.0, 1.0)

    '''
    for _ in range(perturb_steps):
        x_adv2.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv2, None, 0.055), dim=1),
                                       F.softmax(model(x_natural, None, 0.055), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv2])[0]
        x_adv2 = x_adv1.detach() + step_size * torch.sign(grad.detach())
        x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
    '''
    model.train()

    x_adv1 = Variable(torch.clamp(x_adv1, 0.0, 1.0), requires_grad=False)
    # x_adv2 = Variable(torch.clamp(x_adv2, 0.0, 1.0), requires_grad=False)
   
    #x_adv2 = x_natural
    #x_adv2.requires_grad = True
    #outputs = model(x_adv2)
    #loss = F.nll_loss(outputs, y)
    #loss.backward()

    #x_adv2 = fgsm_attack(x_adv2, 8/255.0, x_adv2.grad.data)
    
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    y_adv = model(x_adv1)

    consit_loss = consistancy_loss(y_adv, logits, tem, conf, logger)
    
    #print(loss_robust)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(y_adv, dim=1),
                                                    F.softmax(logits, dim=1))

    #loss_robust += (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv2), dim=1),
    #                                                F.softmax(model(x_natural), dim=1))


    #print(loss_robust, consit_loss, loss_natural)
    #loss = loss_natural + beta * (loss_robust + 0.1*consit_loss)
    loss = loss_natural + beta * (consit_loss)
    return loss

def consistancy_loss(y_adv, logits, tem, conf, logger):

    y_1 = F.softmax(y_adv, dim=1)
    y_2 = F.softmax(logits, dim=1)
    avg_y = (y_1 + y_2)/2

    
    num = torch.pow(avg_y, 1./tem)
    deno = torch.sum(torch.pow(avg_y, 1./tem), dim=1, keepdim=True) + 1e-8
    sharp_p = (num / deno).detach()
    
    loss = 0
    
    loss += torch.mean((-sharp_p * torch.log(y_1+1e-8)).sum(1)[avg_y.max(1)[0] > conf])
    loss += torch.mean((-sharp_p * torch.log(y_2+1e-8)).sum(1)[avg_y.max(1)[0] > conf])
    return loss/2

'''
if __name__ == '__main__':
   output_dir = pathlib.Path('/scratch/ag7644/output/cifar-small-V7/adv/robust-wrt-s3')

   logger = create_logger(name=__name__,
                           distributed_rank=0,
                           output_dir=output_dir,
                           filename='log.txt')

'''
