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

'''
160 - 0.0037
170 - 0.0023
180 - 0.0015
190 - 0.00096
200 - 0.00061
'''
temp = 0.00061
if config.dataset.name == 'CIFAR10':
    train_loader, val_loader = create_cifar_dataloader(config, is_train=True)
else:
    train_loader, val_loader = create_dataloader(config, is_train=True)

#search_model_path = '/home/utkarsh/output/search_cifar_10/checkpoint_00100.pth'
#train_model_path = '/home/utkarsh/output/train_cifar_10/checkpoint_00200.pth'

search_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-l1-sgd/checkpoint_00100.pth'
#train_model_path = '/scratch/un270/output/imagenet-v8/train/r18-adv-1.0/checkpoint_00020.pth'
#train_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth'
#train_model_path = '/scratch/un270/output/cifar-small/adv/robust-s1-t1-5-100/checkpoint_00100.pth'
#train_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-t1-7/checkpoint_00190.pth'
#train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-s5-0-standard/checkpoint_00100.pth" 
#train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-l2-0-standard/checkpoint_00100.pth"
train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-l1-adam-2/checkpoint_00020.pth"
#train_model_path = "/scratch/un270/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
#train_model_path = "/scratch/un270/output/cifar-small-V2/train/robust-s5-0-standard/checkpoint_00100.pth"
#train_model_path = "/scratch/un270/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
#train_model_path = '/scratch/un270/output/cifar-small/search/robust-s1-t1-5/checkpoint_00200.pth'

#train_model_path = '/scratch/un270/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth'
#train_model_path = '/scratch/un270/output/cifar-small/train/robust-s1-l2-sgd-only-arch/checkpoint_00098.pth'

#train_model_path = "/scratch/un270/output/cifar-small-V3/train/robust-s3-s1-t1/checkpoint_00090.pth"

#train_model_path = "/scratch/un270/output/cifar-small-V4/train/robust-s1-t1-7-2/checkpoint_00200.pth"

# All five
train_model_path = '/scratch/ag7644/output/cifar-small/train/robust-s1-t1-3/checkpoint_00200.pth'
#train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-5/checkpoint_00200.pth'
#train_model_path = '/scratch/ag7644/output/cifar-small/search/robust-s1-t1-7/checkpoint_00190.pth'
train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/search/robust-s1-l1-sgd/checkpoint_00100.pth"

#train_model_path = '/scratch/ag7644/output/cifar-small-V4/adv/robust-s3-t1/checkpoint_00020.pth'
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-t1-5/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-t1-7/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-l2-adam-2/checkpoint_00020.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small/adv/robust-s1-l1-adam-2/checkpoint_00020.pth"


#train_model_path = "/scratch/ag7644/output/cifar-small-V4/search/robust-l-standard-run2/checkpoint_00100.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small-V6/train/robust-r50-s3-2/checkpoint_00200.pth"


#### KD varients paths ####
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:resnet8x4_T:resnet32x4_cifar10_similarity_r:1_a:0.0_b:3000.0_1/resnet8x4_best.pth"
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:resnet8x4_T:resnet32x4_cifar10_rkd_r:1_a:0.0_b:1.0_1/resnet8x4_best.pth"
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:resnet8x4_T:resnet32x4_cifar10_vid_r:1_a:0.0_b:1.0_1/resnet8x4_best.pth"
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:resnet8x4_T:resnet32x4_cifar10_factor_r:1_a:0.0_b:200.0_1/resnet8x4_best.pth"

#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:rnas-cl-s5_T:resnet32x4_cifar10_factor_r:1_a:0.0_b:200.0_1/rnas-cl-s5_best.pth"
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:rnas-cl-s5_T:resnet32x4_cifar10_similarity_r:1_a:0.0_b:3000.0_1/rnas-cl-s5_best.pth"
#train_model_path = "/home/ag7644/RepDistiller/save/student_model/S:rnas-cl-s5_T:resnet32x4_cifar10_vid_r:1_a:0.0_b:1.0_1/rnas-cl-s5_best.pth"

#curr_dir = "/home/ag7644/RepDistiller/save/student_model/"

#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_vid_r:1_a:0.0_b:1.0_1/rnas-cl-s7_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_similarity_r:1_a:0.0_b:3000.0_1/rnas-cl-s7_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_rkd_r:1_a:0.0_b:1.0_1/rnas-cl-s7_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_factor_r:1_a:0.0_b:200.0_1/rnas-cl-s7_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_crd_r:1_a:0.0_b:0.8_1/rnas-cl-s7_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s7_T:resnet32x4_cifar10_pkt_r:1_a:0.0_b:30000.0_1/rnas-cl-s7_best.pth"

#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_vid_r:1_a:0.0_b:1.0_2/rnas-cl-s5_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_rkd_r:1_a:0.0_b:1.0_2/rnas-cl-s5_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_factor_r:1_a:0.0_b:200.0_2/rnas-cl-s5_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_crd_r:1_a:0.0_b:0.8_2/rnas-cl-s5_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_similarity_r:1_a:0.0_b:3000.0_1/rnas-cl-s5_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s5_T:resnet32x4_cifar10_pkt_r:1_a:0.0_b:30000.0_1/rnas-cl-s5_best.pth"

#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_factor_r:1_a:0.0_b:200.0_2/rnas-cl-s3_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_crd_r:1_a:0.0_b:0.8_2/rnas-cl-s3_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_similarity_r:1_a:0.0_b:3000.0_2/rnas-cl-s3_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_rkd_r:1_a:0.0_b:1.0_2/rnas-cl-s3_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_vid_r:1_a:0.0_b:1.0_2/rnas-cl-s3_best.pth"
#train_model_path = curr_dir + "S:rnas-cl-s3_T:resnet32x4_cifar10_pkt_r:1_a:0.0_b:30000.0_1/rnas-cl-s3_best.pth"

### No KL, only cross connections
#train_model_path = "/scratch/ag7644/output/cifar-small-V6/train/robust-wrt-s3-no-kl/checkpoint_00200.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small-V6/train/robust-wrt-s5-no-kl/checkpoint_00200.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small-V6/train/robust-wrt-s7-no-kl/checkpoint_00200.pth"

#train_model_path = "/scratch/ag7644/output/cifar-small-V6/train/robust-wrt-l2-no-kl/checkpoint_00200.pth"
#train_model_path = "/scratch/ag7644/output/cifar-small-V6/search/robust-wrt-l1-no-kl/checkpoint_00100.pth"

print(train_model_path)
#l2
# /scratch/un270/output/cifar-small/search/robust-s1-l2-sgd/checkpoint_00100.pth
# /scratch/un270/output/cifar-small/adv/robust-s1-l2-adam-2

#robust-s1-l2-sgd-only-arch
#robust-s1-l1-adam

#### Get trained model ####
#teacher_model = WideResNet()
#teacher_model.load_state_dict(torch.load("robust_models/model_cifar_wrn.pt"))

path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
#model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/cifar_model_weights_30_epochs.pth')
#teacher_model = Fast_AT('cuda', model_path)

model_path = os.path.join(path,'/scratch/ag7644/cifar_robust_checkpoints/cifar10_wide10_linf_eps8.pth')
teacher_model = Robust_Overfitting('cuda', model_path)

#model_path = os.path.join(path,'/home/unath/RKD_NAS/DAS/model/ciafar10_checkpoints/model-wideres-pgdHE-wide10.pt')
#teacher_model = AT_HE(config.device, model_path)
#teacher_model.load()
#teacher_model.eval()

#path_wide_Resnet_2 = "/scratch/un270/wide_resnet50_2_l2_eps1.ckpt"
#dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)
#model = dataset.get_model('wide_resnet50_2', False)
#_, _, teacher_model = load_teacher(model, path_wide_Resnet_2)
#teacher_model = teacher_model.cuda()

teacher_dict = get_weight_dict_from_teacher(teacher_model)
model = create_model(config, teacher_dict)
#model = load_searched_model(model, search_model_path, config.device)
'''
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
    teacher_dict[key] = teacher_dict[key].cuda()
'''
model = create_model(config, teacher_dict)
#model = load_searched_model(model, search_model_path, config.device)
########################

train_model_path = "/scratch/ag7644/output/cifar-small-V7/adv/robust-wrt-s7-consit2/checkpoint_00020.pth"

#model = create_model(config)
#model = load_searched_model(model, search_model_path)
#macs, n_params = count_op(config, model)
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

'''
################### AA #####################
for param in model.parameters():
    param.requires_grad = False
x_test = [x for (x, y) in val_loader]
x_test = torch.cat(x_test, dim=0)
y_test = [y for (x, y) in val_loader]
y_test = torch.cat(y_test, dim=0)
model.eval()

adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
x_adv = adversary.run_standard_evaluation(x_test.cuda(), y_test.cuda(), bs=config.validation.batch_size)
#############################################


print("AUTO DONE")
'''


for i, (images, labels) in enumerate(val_loader):
    #print(i)
    images = images.cuda()
    #all_output, teacher_output = teacher_model(images)
    #mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
    #std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()
    #with torch.no_grad():
    #     all_output, teacher_output = teacher_model(images)
    #images = (images - mu)/std
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
     
    #total_dif_spectrum += spectrum
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    #if i >100:
    #   break

#print(total_dif_spectrum)
print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

#model = WideResNet()
#model.load_state_dict(torch.load("robust_models/model_cifar_wrn.pt"))
'''
path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
model_path = os.path.join(path,'/scratch/un270/ciafar10_robust_checkpoints/model-wideres-pgdHE-wide10.pt')
model = AT_HE(config.device, model_path)
model.load()
model = model.cuda().eval()
'''


model = model.cuda().eval()


## Code for pgd attack

#atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
#atk = torchattacks.FGSM(model, eps=8/255)
#atk  = torchattacks.CW(model, c=1e-4, kappa=0, steps=10, lr=0.01)
#atk = torchattacks.MIFGSM(model, eps=8/255, steps=20, alpha=0.8/255, decay=10.0)
#atk = torchattacks.DeepFool(model, steps=10, overshoot=0.8/255)

'''
for i in range(3,6):
    myeps = 0.01 * (i+1)
    print(myeps)
    myeps = 8/255.0
    atk = PGD(model, eps=myeps, alpha=0.8/255, steps=10)
    atk.set_return_type('int')
    atk.save(data_loader=val_loader, save_path="/scratch/ag7644/output/cifarpgd.pt", verbose=True)
'''



myeps = 8/255
print("eps: ",myeps)
atk = PGD(model, eps=myeps, alpha=0.8/255, steps=10)
#atk = PGDL2(model, eps=1.0, alpha=0.2, steps=7)
atk.set_return_type('int') # Save as integer.

#adv_images, adv_labels = torch.load("/scratch/un270/output/cifar10_pgd.pt")
#mean=torch.tensor([0.491, 0.482, 0.447])
#std=torch.tensor([0.247, 0.243, 0.262])
#adv_images = ((adv_images.float()/255)-mean[None, :, None, None])/std[None, :, None, None]
#adv_data = TensorDataset(adv_images.float()/255, adv_labels)
#val_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

atk.save(data_loader=val_loader, save_path="/scratch/ag7644/output/cifarpgd.pt", verbose=True)

'''
# Load Adversarial Images
adv_images, adv_labels = torch.load("/scratch/ag7644/output/cifarpgd.pt")
#mean=torch.tensor([0.491, 0.482, 0.447])
#std=torch.tensor([0.247, 0.243, 0.262])
#adv_images = ((adv_images.float()/255)-mean[None, :, None, None])/std[None, :, None, None]
#adv_data = TensorDataset(adv_images, adv_labels)
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)
'''

#model_test = WideResNet()
#model_test.load_state_dict(torch.load("robust_models/model_cifar_rkd_nas.pt"))

## Standard Accuracy
#model_test.eval()


## Robust Accuracy
#model_test.eval()

correct = 0
total = 0

mystep = 20

myeps = 8/255.0
print(myeps)
for images, labels in val_loader:
    mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
    std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1)
    images = (images - mu)/std
    
    #images = pgd_whitebox(model, images, labels.cuda(), 'cuda', 8/255, 20, 0.8/255, 0, 1)
    #images = pgd_whitebox(model, images, labels.cuda(), num_steps=mystep)
    #images = cw_whitebox(model, images, labels.cuda())
    
    images = images.cuda() 
    images.requires_grad = True
    outputs = model(images)
    loss = F.nll_loss(outputs, labels.cuda())
    loss.backward()
    
    #images = cw_whitebox(model, images, labels.cuda(), epsilon=myeps)
    images = fgsm_attack(images, myeps, images.grad.data)
    outputs = model(images, None, temp)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
print('Robust PDG accuracy: %.2f %%' % (100 * float(correct) / total))

'''
correct = 0
total = 0

for images, labels in val_loader:
    images = images.cuda()
    #mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).cuda()
    #std = torch.tensor((0.2471, 0.2435, 0.2616)).view(3,1,1).cuda()
    #images = (images - mu)/std
    #images = pgd_whitebox(model, images, labels.cuda(), 'cuda', 8/255, 20, 0.8/255, 0, 1)
    #images = pgd_whitebox(model, images, labels.cuda())
    images = cw_whitebox(model, images, labels.cuda(), num_steps=mystep)
    outputs = model(images, None, temp)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Robust CW accuracy: %.2f %%' % (100 * float(correct) / total))
print("mystep ", mystep)
'''
