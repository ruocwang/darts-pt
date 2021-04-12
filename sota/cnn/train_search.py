import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import random
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

from sota.cnn.model_search import Network as DartsNetwork
from sota.cnn.model_search_sdarts import SDartsNetwork
from sota.cnn.model_search_darts_proj import DartsNetworkProj
from sota.cnn.model_search_sdarts_proj import SDartsNetworkProj
from attacker.perturb import Linf_PGD_alpha, Random_alpha
# from optimizers.darts.architect import Architect as DartsArchitect
from nasbench201.architect_ig import Architect
from sota.cnn.spaces import spaces_dict

from torch.utils.tensorboard import SummaryWriter
from sota.cnn.projection import pt_project


torch.set_printoptions(precision=4, sci_mode=False)

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
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
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
#### common
parser.add_argument('--ckpt_interval', type=int, default=10, help="interval (epoch) for saving checkpoints")
parser.add_argument('--method', type=str)
parser.add_argument('--arch_opt', type=str, default='adam', help='architecture optimizer')
parser.add_argument('--resume_epoch', type=int, default=0, help="load ckpt, start training at resume_epoch")
parser.add_argument('--resume_expid', type=str, default='', help="full expid to resume from, name == ckpt folder name")
parser.add_argument('--dev', type=str, default='', help="dev mode")
parser.add_argument('--deter', action='store_true', default=False, help='fully deterministic, for debugging only, slow like hell')
parser.add_argument('--expid_tag', type=str, default='', help="extra tag for expid, 'debug' for debugging")
parser.add_argument('--log_tag', type=str, default='', help="extra tag for log, use 'debug' for debug")
#### darts 2nd order
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
#### sdarts
parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
#### dev
## common
parser.add_argument('--tune_epochs', type=int, default=140, help='not used for projection (use proj_intv instead)')
parser.add_argument('--fast', action='store_true', default=False, help='eval/train on one batch, for debugging')
parser.add_argument('--dev_resume_epoch', type=int, default=-1, help="resume epoch for arch selection phase, starting from 0")
parser.add_argument('--dev_resume_log', type=str, default='', help="resume log name for arch selection phase")
## projection
parser.add_argument('--edge_decision', type=str, default='sgas', choices=['random'], help='used for both proj_op and proj_edge')
parser.add_argument('--proj_crit_normal', type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_crit_reduce', type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_crit_edge',   type=str, default='acc', choices=['loss', 'acc'])
parser.add_argument('--proj_intv', type=int, default=10, help='interval between two projections')
parser.add_argument('--proj_mode_edge', type=str, default='reg', choices=['reg'],
                    help='edge projection evaluation mode, reg: one edge at a time')

args = parser.parse_args()

#### macros


#### args augment
if args.expid_tag != '':
    args.save += '-{}'.format(args.expid_tag)

expid = args.save
args.save = '../../experiments/sota/{}/search-{}-{}-{}'.format(
    args.dataset, args.save, args.search_space, args.seed)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)

if args.resume_epoch > 0: # do not delete dir when resume:
    args.save = '../../experiments/sota/{}/{}'.format(args.dataset, args.resume_expid)
    assert(os.path.exists(args.save), 'resume but {} does not exist!'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py') + glob.glob('../../nasbench201/architect*.py') + glob.glob('../../optimizers/darts/architect.py')
    if os.path.exists(args.save):
        if 'debug' in args.expid_tag or input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)


#### logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log'
if args.resume_epoch > 0:
    log_file += '_resume-{}'.format(args.resume_epoch)
if args.dev_resume_epoch >= 0:
    log_file += '_dev-resume-{}'.format(args.dev_resume_epoch)
if args.dev != '':
    log_file += '_dev-{}'.format(args.dev)
    if args.dev == 'proj':
        log_file += '_intv-{}_ED-{}_PCN-{}_PCR-{}'.format(
                    args.proj_intv, args.edge_decision, args.proj_crit_normal, args.proj_crit_reduce)
    else:
        print('ERROR: DEV METHOD NOT SUPPORTED IN LOGGING:', args.dev); exit(0)
    log_file += '_seed-{}'.format(args.seed)

    if args.log_tag != '': log_file += '_tag-{}'.format(args.log_tag)
if args.log_tag == 'debug': ## prevent redundant debug log files
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

#### dev resume dir
args.dev_resume_checkpoint_dir = os.path.join(args.save, args.dev_resume_log)
print(args.dev_resume_checkpoint_dir)
if not os.path.exists(args.dev_resume_checkpoint_dir):
    os.mkdir(args.dev_resume_checkpoint_dir)
args.dev_save_checkpoint_dir = os.path.join(args.save, log_file.replace('.txt', ''))
print(args.dev_save_checkpoint_dir)
if not os.path.exists(args.dev_save_checkpoint_dir):
    os.mkdir(args.dev_save_checkpoint_dir)

if args.dataset == 'cifar100':
    n_classes = 100
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
    logging.info('gpu device = %d' % gpu)
    logging.info("args = %s", args)

    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = ig_utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

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

    test_queue  = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        pin_memory=True)


    #### sdarts
    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha
    else:
        print('ERROR PERTURB_ALPHA TYPE:', args.perturb_alpha); exit(1)
    
    #### model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    ## darts
    if args.method in ['darts', 'blank']:
        model = DartsNetwork(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    ## sdarts
    elif args.method == 'sdarts':
        model = SDartsNetwork(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    ## projection
    elif args.method in ['darts-proj', 'blank-proj']:
        model = DartsNetworkProj(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    elif args.method in ['sdarts-proj']:
        model = SDartsNetworkProj(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    else:
        print('ERROR: WRONG MODEL:', args.method); exit(0)
    model = model.cuda()

    architect = Architect(model, args)

    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))


    #### scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    #### resume
    start_epoch = 0
    if args.resume_epoch > 0:
        logging.info('loading checkpoint from {}'.format(expid))
        filename = os.path.join(args.save, 'checkpoint_{}.pth.tar'.format(args.resume_epoch))

        if os.path.isfile(filename):
            logging.info("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            resume_epoch = checkpoint['epoch'] # epoch
            model.load_state_dict(checkpoint['state_dict']) # model
            saved_arch_parameters = checkpoint['alpha'] # arch
            model.set_arch_parameters(saved_arch_parameters)
            scheduler.load_state_dict(checkpoint['scheduler'])
            model.optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
            architect.optimizer.load_state_dict(checkpoint['arch_optimizer']) # arch optimizer
            start_epoch = args.resume_epoch
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(filename, resume_epoch))
        else:
            logging.info("=> no checkpoint found at '{}'".format(filename))


    #### main search
    logging.info('starting training at epoch {}'.format(start_epoch))
    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_lr()[0]

        ## data aug
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr, train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        
        ## sdarts
        if args.perturb_alpha:
            epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        ## logging
        num_params = ig_utils.count_parameters_in_Compact(model)
        genotype = model.genotype()
        logging.info('param size = %f', num_params)
        logging.info('genotype = %s', genotype)
        model.printing(logging)

        ## training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, model.optimizer, lr, epoch,
                                     perturb_alpha, epsilon_alpha)
        logging.info('train_acc %f | train_obj %f', train_acc, train_obj)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)

        ## scheduler updates (before saving ckpts)
        scheduler.step()

        ## validation
        valid_acc, valid_obj = infer(valid_queue, model, log=False)
        logging.info('valid_acc %f | valid_obj %f', valid_acc, valid_obj)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)

        test_acc, test_obj = infer(test_queue, model, log=False)
        logging.info('test_acc %f | test_obj %f', test_acc, test_obj)
        writer.add_scalar('Acc/test', test_acc, epoch)
        writer.add_scalar('Obj/test', test_obj, epoch)

        ## saving
        if (epoch + 1) % args.ckpt_interval == 0:
            save_state_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'alpha': model.arch_parameters(),
                'optimizer': model.optimizer.state_dict(),
                'arch_optimizer': architect.optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            ig_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=True)

    #### projection
    if args.dev == 'proj':
        pt_project(train_queue, valid_queue, model, architect, model.optimizer,
                   start_epoch, args, infer, perturb_alpha, args.epsilon_alpha)

    writer.close()


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch,
          perturb_alpha, epsilon_alpha):
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
        architect.step(input, target, input_search, target_search, lr, optimizer)

        ## sdarts
        if perturb_alpha:
            # transform arch_parameters to prob (for perturbation)
            model.softmax_arch_parameters()
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            perturb_alpha(model, input, target, epsilon_alpha)

        ## train weights
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args)
        
        ## sdarts
        if perturb_alpha:
            ## restore alpha to unperturbed arch_parameters
            model.restore_arch_parameters()

        ## logging
        n = input.size(0)
        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        if args.fast:
            break

    return  top1.avg, objs.avg


def infer(valid_queue, model, log=True, _eval=True, weights_dict=None):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if _eval else model.train() # disable running stats for projection

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            
            if weights_dict is None:
                loss, logits = model._loss(input, target, return_logits=True)
            else:
                logits = model(input, weights_dict=weights_dict)
                loss = model._criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0 and log:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            if args.fast:
                break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()