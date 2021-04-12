import os
import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import nasbench201.utils as ig_utils
import logging
import torch.utils

from copy import deepcopy

torch.set_printoptions(precision=4, sci_mode=False)


def project_op(model, proj_queue, args, infer, cell_type, selected_eid=None):
    ''' operation '''
    #### macros
    num_edges, num_ops = model.num_edges, model.num_ops
    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = args.proj_crit[cell_type]
    
    #### select an edge
    if selected_eid is None:
        remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        if args.edge_decision == "random":
            selected_eid = np.random.choice(remain_eids, size=1)[0]
            logging.info('selected edge: %d %s', selected_eid, cell_type)

    #### select the best operation
    if proj_crit == 'loss':
        crit_idx = 1
        compare = lambda x, y: x > y
    elif proj_crit == 'acc':
        crit_idx = 0
        compare = lambda x, y: x < y

    best_opid = 0
    crit_extrema = None
    for opid in range(num_ops):
        ## projection
        weights = model.get_projected_weights(cell_type)
        proj_mask = torch.ones_like(weights[selected_eid])
        proj_mask[opid] = 0
        weights[selected_eid] = weights[selected_eid] * proj_mask

        ## proj evaluation
        weights_dict = {cell_type:weights}
        valid_stats = infer(proj_queue, model, log=False, _eval=False, weights_dict=weights_dict)
        crit = valid_stats[crit_idx]

        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_opid = opid
        logging.info('valid_acc  %f', valid_stats[0])
        logging.info('valid_loss %f', valid_stats[1])

    #### project
    logging.info('best opid: %d', best_opid)
    return selected_eid, best_opid
    

def project_edge(model, proj_queue, args, infer, cell_type):
    ''' topology '''
    #### macros
    candidate_flags = model.candidate_flags_edge[cell_type]
    proj_crit = args.proj_crit[cell_type]

    #### select an edge
    remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
    if args.edge_decision == "random":
        selected_nid = np.random.choice(remain_nids, size=1)[0]
        logging.info('selected node: %d %s', selected_nid, cell_type)
    
    #### select top2 edges
    if proj_crit == 'loss':
        crit_idx = 1
        compare = lambda x, y: x > y
    elif proj_crit == 'acc':
        crit_idx = 0
        compare = lambda x, y: x < y

    eids = deepcopy(model.nid2eids[selected_nid])
    while len(eids) > 2:
        eid_todel = None
        crit_extrema = None
        for eid in eids:
            weights = model.get_projected_weights(cell_type)
            weights[eid].data.fill_(0)
            weights_dict = {cell_type:weights}

            ## proj evaluation
            valid_stats = infer(proj_queue, model, log=False, _eval=False, weights_dict=weights_dict)
            crit = valid_stats[crit_idx]

            if crit_extrema is None or not compare(crit, crit_extrema): # find out bad edges
                crit_extrema = crit
                eid_todel = eid
            logging.info('valid_acc %f', valid_stats[0])
            logging.info('valid_loss %f', valid_stats[1])
        eids.remove(eid_todel)

    #### project
    logging.info('top2 edges: (%d, %d)', eids[0], eids[1])
    return selected_nid, eids


def pt_project(train_queue, valid_queue, model, architect, optimizer,
               epoch, args, infer, perturb_alpha, epsilon_alpha):
    model.train()
    model.printing(logging)

    train_acc, train_obj = infer(train_queue, model, log=False)
    logging.info('train_acc  %f', train_acc)
    logging.info('train_loss %f', train_obj)

    valid_acc, valid_obj = infer(valid_queue, model, log=False)
    logging.info('valid_acc  %f', valid_acc)
    logging.info('valid_loss %f', valid_obj)

    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()


    #### macros
    num_projs = model.num_edges + len(model.nid2eids.keys()) - 1 ## -1 because we project at both epoch 0 and -1
    tune_epochs = args.proj_intv * num_projs + 1
    proj_intv = args.proj_intv
    args.proj_crit = {'normal':args.proj_crit_normal, 'reduce':args.proj_crit_reduce}
    proj_queue = valid_queue


    #### reset optimizer
    model.reset_optimizer(args.learning_rate / 10, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, float(tune_epochs), eta_min=args.learning_rate_min)


    #### load proj checkpoints
    start_epoch = 0
    if args.dev_resume_epoch >= 0:
        filename = os.path.join(args.dev_resume_checkpoint_dir, 'checkpoint_{}.pth.tar'.format(args.dev_resume_epoch))
        if os.path.isfile(filename):
            logging.info("=> loading projection checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.set_state_dict(architect, scheduler, checkpoint)
            model.set_arch_parameters(checkpoint['alpha'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            model.optimizer.load_state_dict(checkpoint['optimizer']) # optimizer
        else:
            logging.info("=> no checkpoint found at '{}'".format(filename))
            exit(0)


    #### projecting and tuning
    for epoch in range(start_epoch, tune_epochs):
        logging.info('epoch %d', epoch)
        
        ## project
        if epoch % proj_intv == 0 or epoch == tune_epochs - 1:
            ## saving every projection
            save_state_dict = model.get_state_dict(epoch, architect, scheduler)
            ig_utils.save_checkpoint(save_state_dict, False, args.dev_save_checkpoint_dir, per_epoch=True)

            if epoch < proj_intv * model.num_edges:
                logging.info('project op')
                
                selected_eid_normal, best_opid_normal = project_op(model, proj_queue, args, infer, cell_type='normal')
                model.project_op(selected_eid_normal, best_opid_normal, cell_type='normal')
                selected_eid_reduce, best_opid_reduce = project_op(model, proj_queue, args, infer, cell_type='reduce')
                model.project_op(selected_eid_reduce, best_opid_reduce, cell_type='reduce')

                model.printing(logging)
            else:
                logging.info('project edge')
                
                selected_nid_normal, eids_normal = project_edge(model, proj_queue, args, infer, cell_type='normal')
                model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
                selected_nid_reduce, eids_reduce = project_edge(model, proj_queue, args, infer, cell_type='reduce')
                model.project_edge(selected_nid_reduce, eids_reduce, cell_type='reduce')

                model.printing(logging)

        ## tune
        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            ## fetch data
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            ## train alpha
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            architect.step(input, target, input_search, target_search,
                           return_logits=True)

            ## sdarts
            if perturb_alpha:
                # transform arch_parameters to prob (for perturbation)
                model.softmax_arch_parameters()
                optimizer.zero_grad(); architect.optimizer.zero_grad()
                perturb_alpha(model, input, target, epsilon_alpha)

            ## train weight
            optimizer.zero_grad(); architect.optimizer.zero_grad()
            logits, loss = model.step(input, target, args)

            ## sdarts
            if perturb_alpha:
                ## restore alpha to unperturbed arch_parameters
                model.restore_arch_parameters()

            ## logging
            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            if args.fast:
                break

        ## one epoch end
        model.printing(logging)

        train_acc, train_obj = infer(train_queue, model, log=False)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, log=False)
        logging.info('valid_acc  %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)


    logging.info('projection finished')
    model.printing(logging)
    num_params = ig_utils.count_parameters_in_Compact(model)
    genotype = model.genotype()
    logging.info('param size = %f', num_params)
    logging.info('genotype = %s', genotype)

    return