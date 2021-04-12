import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from nasbench201.utils import drop_path


class MixedOp(nn.Module):
    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        ret = sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)
        return ret


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, drop_prob=0.):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if drop_prob > 0. and self.training:
                s = sum(drop_path(self._ops[offset+j](h, weights[offset+j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, primitives, args,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0):
        super(Network, self).__init__()
        #### original code
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob

        nn.Module.PRIMITIVES = primitives; self.op_names = primitives

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

        #### optimizer
        self._args = args
        self.optimizer = torch.optim.SGD(
            self.get_weights(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        

    def reset_optimizer(self, lr, momentum, weight_decay):
        del self.optimizer
        self.optimizer = torch.optim.SGD(
            self.get_weights(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay)

    def _loss(self, input, target, return_logits=False):
        logits = self(input)
        loss = self._criterion(logits, target)
        return (loss, logits) if return_logits else loss

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])
        self.num_edges = k
        self.num_ops = num_ops

        self.alphas_normal = self._initialize_alphas_numpy(k, num_ops)
        self.alphas_reduce = self._initialize_alphas_numpy(k, num_ops)
        self._arch_parameters = [ # must be in this order!
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def _initialize_alphas_numpy(self, k, num_ops):
        ''' init from specified arch '''
        return Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)

    def forward(self, input):
        weights = self.get_softmax()
        weights_normal = weights['normal']
        weights_reduce = weights['reduce']

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal

            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))

        return logits

    def step(self, input, target, args, shared=None):
        assert shared is None, 'gradient sharing disabled'
        
        Lt, logit_t = self._loss(input, target, return_logits=True)
        Lt.backward()

        nn.utils.clip_grad_norm_(self.get_weights(), args.grad_clip)
        self.optimizer.step()

        return logit_t, Lt

    #### utils
    def set_arch_parameters(self, new_alphas):
        for alpha, new_alpha in zip(self.arch_parameters(), new_alphas):
            alpha.data.copy_(new_alpha.data)

    def get_softmax(self):
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        return {'normal':weights_normal, 'reduce':weights_reduce}

    def printing(self, logging, option='all'):
        weights = self.get_softmax()
        if option in ['all', 'normal']:
            weights_normal = weights['normal']
            logging.info(weights_normal)
        if option in ['all', 'reduce']:
            weights_reduce = weights['reduce']
            logging.info(weights_reduce)

    def arch_parameters(self):
        return self._arch_parameters

    def get_weights(self):
        return self.parameters()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, self._args,\
                            drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct'] ## two are equal for Darts space

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError: # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
