import torch
from torch import tensor, nn
from torchvision import models
from train import load_config
from robustness import datasets
from robustness.tools import constants, helpers
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchattacks import PGD

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
config = load_config()
set_seed(config)

train_loader, val_loader = create_dataloader(config, is_train=True)

def print_model(model):
    count=0
    for k,v in model.state_dict().items():
        if len(v.shape)==4:
           count+=1
           #print(k, v.shape)
    print(count)


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


path_resnet18 = "/home/unath/pretrained-models/resnet-18-l2-eps3.ckpt"
#path_wide_Resnet_2 = "/home/unath/wide_resnet50_2_linf_eps8.0.ckpt"
path_wide_Resnet_2 = "/home/utkarsh/wide_resnet50_2_l2_eps1.ckpt"

dataset = datasets.DATASETS['imagenet'](config.dataset.dataset_dir, num_classes=1000)

#train_loader, val_loader = dataset.make_loaders(8, 32, True)

#train_loader = helpers.DataPrefetcher(train_loader)
#val_loader = helpers.DataPrefetcher(val_loader)

model = dataset.get_model('wide_resnet50_2', False)
print_model(model)
#model = models.wide_resnet50_2()
mean, std, model = load_searched_model(model, path_wide_Resnet_2)

#mean, std, model = load_searched_model(model, args.resume)
model.cuda()
model.eval()
mean.cuda()
std.cuda()


correct = 0.0
total = 0.0


writer = SummaryWriter()
iterator = tqdm(enumerate(val_loader), total=len(val_loader))
for i, (images, labels) in iterator:
    images = images.cuda()
    with torch.no_grad():
          x = torch.clamp(images, 0, 1)
          #normalized_inp = (x - mean)/std
          normalized_inp = images
          outputs = model(normalized_inp)
          #outputs = outputs[:,[15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]]
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #print(predicted, labels)
    correct += (predicted == labels.cuda()).sum()
    desc = (' Epoch:{0} | Loss {1} | '
                '{2}1  ||'.format( i, total, correct / total))
    iterator.set_description(desc)
    iterator.refresh()
    writer.add_scalar("Loss/train", correct, total, correct / total)
    
writer.flush()
print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

model = model.cuda().eval()

atk = PGD(model, eps=1/255, alpha=2/255, steps=7)

atk.set_return_type('int') # Save as integer.
atk.save(data_loader=val_loader, save_path="/home/utkarsh/output/pgd.pt", verbose=True)

# Load Adversarial Images
adv_images, adv_labels = torch.load("/home/utkarsh/output/pgd.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=32, shuffle=False)


correct = 0
total = 0

for images, labels in adv_loader:
    images = images.cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
