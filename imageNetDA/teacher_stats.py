import numpy as np
import torch
import torch.nn as nn

import torchattacks
from torchattacks import PGD, PGDL2

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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

from myattack import cw_whitebox , pgd_whitebox, fgsm_attack
from autoattack import AutoAttack

from PIL import Image
import matplotlib.pyplot as plt

temp = 0.00061
if config.dataset.name == 'CIFAR10':
    train_loader, val_loader = create_cifar_dataloader(config, is_train=True)
else:
    train_loader, val_loader = create_dataloader(config, is_train=True)


# All five
#train_model_path = '/scratch/ag7644/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth'
#train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-5/checkpoint_00200.pth'
#train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-7/checkpoint_00190.pth'
#train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l1-sgd/checkpoint_00100.pth"

#train_model_path = '/scratch/ag7644/output/cifar-small-V4/adv/robust-s3-t1/checkpoint_00020.pth'
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-t1-5/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-t1-7/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-l2-adam-2/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-l1-adam-2/checkpoint_00020.pth"

train_model_path = "/scratch/ag7644/output/cifar-small-V5/search/robust-s3/checkpoint_00080.pth"

path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

model_path = os.path.join(path,'/scratch/ag7644/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
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


correct = 0
total = 0

model_path = os.path.join(path,'/scratch/ag7644/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
teacher_model = Robust_Overfitting('cuda', model_path)
teacher_model.load()
teacher_model.eval()
total_dif_spectrum = 0

atk = PGD(teacher_model, eps=8/255, alpha=0.8/255, steps=10)
atk.set_return_type('int')
atk.save(data_loader=val_loader, save_path="/scratch/ag7644/output/cifar_teacher_pgd.pt", verbose=True)

adv_images, adv_labels = torch.load("/scratch/ag7644/output/cifar_teacher_pgd.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

for i, (images, labels) in enumerate(val_loader):
    images = images.cuda()
    all_output, outputs = teacher_model(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

'''
################ Get atention map from teacher model #############
for i, (images, labels) in enumerate(val_loader):
    images = images.cuda()
    all_output, outputs = teacher_model(images)
    for j in range(all_output.shape[1]):
        #if j<11:
        #    continue
        torch.save(all_output, "all_output_0_14.pt")
        break
        os.mkdir("/scratch/ag7644/cifar_teacher_map1/"+str(j)+"-"+str(int(labels[j].data)))
        for k in range(all_output.shape[0]):
            plt.imshow(all_output[k][j].cpu().detach().numpy(), interpolation='bicubic')
            plt.savefig("/scratch/ag7644/cifar_teacher_map1/"+str(j)+"-"+str(int(labels[j].data))+"/"+str(k)+".png")
    break
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
'''


'''
##################### Get connected teacher layer ################
q = np.zeros(31)
mylist = []
for k, v in model.state_dict().items():
    if(k.find("kd_GS_thetas")!=-1): 
       soft_mask_variables = nn.functional.gumbel_softmax(v, 0.013, hard=True)
       index = torch.argmax(soft_mask_variables)
       q[index.data]+=1
       mylist.append(int(index.cpu()))
       print(k, index)

print(q)
print(mylist)
'''
