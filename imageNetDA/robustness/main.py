"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""

from argparse import ArgumentParser
import os
import git
import torch as ch
from torchvision import models

import cox
import cox.utils
import cox.store
from tqdm import tqdm as tqdm
try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults, __version__
    from .defaults import check_and_fill_args
except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

def load_searched_model(model, path):
    state_dict = ch.load(path)['model']
    for k,v in state_dict.items():
        key = k[7:]
        if key=='normalizer.new_mean':
           mean = v
        if key=='normalizer.new_std':
           std = v
        if key.split('.')[0]=='model':
           model.state_dict()[key[6:]].copy_(v)
    return mean, std, model

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # model = models.wide_resnet50_2()
    
    # model = dataset.get_model(args.arch, args.pytorch_pretrained)
    # mean, std, model = load_searched_model(model, args.resume)
    # model.cuda()
    # mean.cuda()
    # std.cuda()

    # correct = 0
    # total = 0

    '''
    for images, labels in val_loader:

       images = images.cuda()
       x = ch.clamp(images, 0, 1)
       x = (x - mean)/std
       outputs = model(x)

       _, predicted = ch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels.cuda()).sum()

    print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
    '''

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module
    '''
    model.eval()
    writer = store.tensorboard
    iterator = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, labels) in iterator:

       images = images.cuda()
       with ch.no_grad():
          x = ch.clamp(images, 0, 1)
          normalized_inp = (x - mean)/std
          #normalized_inp = model.normalizer(images)
          #outputs = model.model(normalized_inp) 
          outputs = model(normalized_inp)

       _, predicted = ch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels.cuda()).sum()
       desc = (' Epoch:{0} | Loss {1} | '
                '{2}1  ||'.format( i, total, correct / total))
       iterator.set_description(desc)
       iterator.refresh() 
       writer.add_scalar("Loss/train", correct, total, correct / total)
       #if total>10000:
       #   break
    writer.flush()
    print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
    '''    
    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)
