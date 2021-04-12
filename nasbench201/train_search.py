import os
import sys
sys.path.insert(0, '../')
import time
import glob
import numpy as np
import torch
import shutil
import nasbench201.utils as ig_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from nasbench201.search_model_darts import TinyNetworkDarts
from nasbench201.search_model_darts_proj import TinyNetworkDartsProj
from nasbench201.cell_operations import SearchSpaceNames
from nasbench201.architect_ig import Architect
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API
from nasbench201.projection import pt_project

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120'], help='choose dataset')
parser.add_argument('--method', type=str, default='dirichlet', help='choose nas method')
parser.add_argument('--search_space', type=str, default='nas-bench-201')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#### common
parser.add_argument('--fast', action='store_true', default=False, help='skip loading api which is slow')
parser.add_argument('--resume_epoch', type=int, default=0, help='0: from scratch; -1: resume from latest checkpoints')
parser.add_argument('--resume_expid', type=str, default='', help='e.g. search-darts-201-2')
parser.add_argument('--dev', type=str, default=None, help='separate supernet traininig and projection phases')
parser.add_argument('--ckpt_interval', type=int, default=20, help='frequency for ckpting')
parser.add_argument('--expid_tag', type=str, default='none', help='extra tag for exp identification')
parser.add_argument('--log_tag', type=str, default='', help='extra tag for log during arch projection')
#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default='acc', choices=['loss', 'acc'], help='criteria for projection')
parser.add_argument('--proj_intv', type=int, default=5, help='fine tune epochs between two projections')
args = parser.parse_args()


#### macros


#### args augment
expid = args.save
args.save = '../experiments/nasbench201/search-{}-{}'.format(args.save, args.seed)
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if args.expid_tag != 'none': args.save += '-' + args.expid_tag


#### logging
if args.resume_epoch > 0: # do not delete dir if resume:
    args.save = '../experiments/nasbench201/{}'.format(args.resume_expid)
    if not os.path.exists(args.save):
        print('no such directory {}'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py') + ['../exp_scripts/{}.sh'.format(expid)]
    if os.path.exists(args.save):
        if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.resume_epoch > 0:
    log_file = 'log_resume-{}_dev-{}_seed-{}_intv-{}'.format(args.resume_epoch, args.dev, args.seed, args.proj_intv)
    if args.log_tag != '':
        log_file += args.log_tag
else:
    log_file = 'log'
if args.log_tag == 'debug':
    log_file = 'log_debug'
log_file += '.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)

if args.log_tag != 'debug' and os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else:
        exit(0)

fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')


#### macros
if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info('gpu device = %d' % gpu)

    if not args.fast:
        api = API('../data/NAS-Bench-201-v1_0-e61699.pth')


    #### model
    criterion = nn.CrossEntropyLoss()
    search_space = SearchSpaceNames[args.search_space]
    if args.method in ['darts', 'blank']:
        model = TinyNetworkDarts(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    elif args.method in ['darts-proj', 'blank-proj']:
        model = TinyNetworkDartsProj(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    model = model.cuda()
    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))

    architect = Architect(model, args)


    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from nasbench201.DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)


    #### scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    #### resume
    start_epoch = 0
    if args.resume_epoch != 0:
        logging.info('loading checkpoint from {}'.format(expid))
        file = 'checkpoint.pth.tar' if args.resume_epoch == -1 else 'checkpoint_{}.pth.tar'.format(args.resume_epoch)
        filename = os.path.join(args.save, file)

        if os.path.isfile(filename):
            logging.info("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            start_epoch = checkpoint['epoch'] # epoch
            model_state_dict = checkpoint['state_dict']
            if '_arch_parameters' in model_state_dict: del model_state_dict['_arch_parameters']
            model.load_state_dict(model_state_dict) # model
            saved_arch_parameters = checkpoint['alpha'] # arch
            model.set_arch_parameters(saved_arch_parameters)
            scheduler.load_state_dict(checkpoint['scheduler'])
            model.optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
            architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(filename, start_epoch - 1))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    #### training
    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_lr()[0]
        ## data aug
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)

        ## pre logging
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        model.printing(logging)

        ## training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, model.optimizer, lr, epoch)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        ## eval
        valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False)
        logging.info('valid_acc  %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)

        ## logging
        if not args.fast:
            # nasbench201
            cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = query(api, model.genotype(), logging)

            # tensorboard
            writer.add_scalars('accuracy', {'train':train_acc,'valid':valid_acc}, epoch)
            writer.add_scalars('loss', {'train':train_obj,'valid':valid_obj}, epoch)
            writer.add_scalars('nasbench201/cifar10', {'train':cifar10_train,'test':cifar10_test}, epoch)
            writer.add_scalars('nasbench201/cifar100', {'train':cifar100_train,'valid':cifar100_valid, 'test':cifar100_test}, epoch)
            writer.add_scalars('nasbench201/imagenet16', {'train':imagenet16_train,'valid':imagenet16_valid, 'test':imagenet16_test}, epoch)

        #### scheduling
        scheduler.step()

        #### saving
        save_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'alpha': model.arch_parameters(),
            'optimizer': model.optimizer.state_dict(),
            'arch_optimizer': architect.optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if save_state['epoch'] % args.ckpt_interval == 0:
            ig_utils.save_checkpoint(save_state, False, args.save, per_epoch=True)

    #### architecture selection / projection
    if args.dev == 'proj':
        pt_project(train_queue, valid_queue, model, architect, criterion, model.optimizer,
                   start_epoch, args, infer, query)

    writer.close()


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()


    for step in range(len(train_queue)):
        model.train()

        ## data
        input, target = next(iter(train_queue))
        input = input.cuda(); target = target.cuda(non_blocking=True)
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda(); target_search = target_search.cuda(non_blocking=True)

        ## train alpha
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        shared = architect.step(input, target, input_search, target_search,
                                eta=lr, network_optimizer=optimizer)

        ## train weight
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args, shared=shared)

        ## logging
        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        if args.fast:
            break

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion,
          log=True, eval=True, weights=None, double=False, bn_est=False):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if eval else model.train() # disable running stats for projection
    
    if bn_est:
        _data_loader = deepcopy(valid_queue)
        for step, (input, target) in enumerate(_data_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
        model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if double:
                input = input.double(); target = target.double()
            
            logits = model(input) if weights is None else model(input, weights=weights)
            loss = criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            if args.fast:
                break

    return top1.avg, objs.avg


#### util functions
def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def query(api, genotype, logging):
    result = api.query_by_arch(genotype)
    logging.info('{:}'.format(result))
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
    logging.info('cifar10 train %f test %f', cifar10_train, cifar10_test)
    logging.info('cifar100 train %f valid %f test %f', cifar100_train, cifar100_valid, cifar100_test)
    logging.info('imagenet16 train %f valid %f test %f', imagenet16_train, imagenet16_valid, imagenet16_test)
    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    main()
