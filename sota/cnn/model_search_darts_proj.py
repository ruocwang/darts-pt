import torch
from copy import deepcopy

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from sota.cnn.model_search import Network

class DartsNetworkProj(Network):
    def __init__(self, C, num_classes, layers, criterion, primitives, args,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(DartsNetworkProj, self).__init__(C, num_classes, layers, criterion, primitives, args,
              steps=steps, multiplier=multiplier, stem_multiplier=stem_multiplier, drop_path_prob=drop_path_prob)
        
        self._initialize_flags()
        self._initialize_proj_weights()
        self._initialize_topology_dicts()

    #### proj flags
    def _initialize_topology_dicts(self):
        self.nid2eids = {0:[2,3,4], 1:[5,6,7,8], 2:[9,10,11,12,13]}
        self.nid2selected_eids = {
            'normal': {0:[],1:[],2:[]},
            'reduce': {0:[],1:[],2:[]},
        }
    
    def _initialize_flags(self):
        self.candidate_flags = {
            'normal':torch.tensor(self.num_edges * [True], requires_grad=False, dtype=torch.bool).cuda(),
            'reduce':torch.tensor(self.num_edges * [True], requires_grad=False, dtype=torch.bool).cuda(),
        } # must be in this order
        self.candidate_flags_edge = {
            'normal': torch.tensor(3 * [True], requires_grad=False, dtype=torch.bool).cuda(),
            'reduce': torch.tensor(3 * [True], requires_grad=False, dtype=torch.bool).cuda(),
        }

    def _initialize_proj_weights(self):
        ''' data structures used for proj '''
        if isinstance(self.alphas_normal, list):
            alphas_normal = torch.stack(self.alphas_normal, dim=0)
            alphas_reduce = torch.stack(self.alphas_reduce, dim=0)
        else:
            alphas_normal = self.alphas_normal
            alphas_reduce = self.alphas_reduce

        self.proj_weights = { # for hard/soft assignment after project
            'normal': torch.zeros_like(alphas_normal),
            'reduce': torch.zeros_like(alphas_reduce),
        }
    
    #### proj function
    def project_op(self, eid, opid, cell_type):
        self.proj_weights[cell_type][eid][opid] = 1 ## hard by default
        self.candidate_flags[cell_type][eid] = False
        
    def project_edge(self, nid, eids, cell_type):
        for eid in self.nid2eids[nid]:
            if eid not in eids: # not top2
                self.proj_weights[cell_type][eid].data.fill_(0)
        self.nid2selected_eids[cell_type][nid] = deepcopy(eids)
        self.candidate_flags_edge[cell_type][nid] = False

    #### critical function
    def get_projected_weights(self, cell_type):
        ''' used in forward and genotype '''
        weights = self.get_softmax()[cell_type]

        ## proj op
        for eid in range(self.num_edges):
            if not self.candidate_flags[cell_type][eid]:
                weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        ## proj edge
        for nid in self.nid2eids:
            if not self.candidate_flags_edge[cell_type][nid]: ## projected node
                for eid in self.nid2eids[nid]:
                    if eid not in self.nid2selected_eids[cell_type][nid]:
                        weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        return weights

    def forward(self, input, weights_dict=None):
        if weights_dict is None or 'normal' not in weights_dict:
            weights_normal = self.get_projected_weights('normal')
        else:
            weights_normal = weights_dict['normal']
        if weights_dict is None or 'reduce' not in weights_dict:
            weights_reduce = self.get_projected_weights('reduce')
        else:
            weights_reduce = weights_dict['reduce']

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

    #### utils
    def printing(self, logging, option='all'):
        weights_normal = self.get_projected_weights('normal')
        weights_reduce = self.get_projected_weights('reduce')

        if option in ['all', 'normal']:
            logging.info('\n%s', weights_normal)
        if option in ['all', 'reduce']:
            logging.info('\n%s', weights_reduce)

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError:
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

        weights_normal = self.get_projected_weights('normal')
        weights_reduce = self.get_projected_weights('reduce')
        gene_normal = _parse(weights_normal.data.cpu().numpy(), True)
        gene_reduce = _parse(weights_reduce.data.cpu().numpy(), False)

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
    
    def get_state_dict(self, epoch, architect, scheduler):
        model_state_dict = {
            'epoch': epoch, ## no +1 because we are saving before projection / at the beginning of an epoch
            'state_dict': self.state_dict(),
            'alpha': self.arch_parameters(),
            'optimizer': self.optimizer.state_dict(),
            'arch_optimizer': architect.optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            #### projection
            'nid2eids': self.nid2eids,
            'nid2selected_eids': self.nid2selected_eids,
            'candidate_flags': self.candidate_flags,
            'candidate_flags_edge': self.candidate_flags_edge,
            'proj_weights': self.proj_weights,
        }
        return model_state_dict

    def set_state_dict(self, architect, scheduler, checkpoint):
        #### common
        self.load_state_dict(checkpoint['state_dict'])
        self.set_arch_parameters(checkpoint['alpha'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        #### projection
        self.nid2eids = checkpoint['nid2eids']
        self.nid2selected_eids = checkpoint['nid2selected_eids']
        self.candidate_flags = checkpoint['candidate_flags']
        self.candidate_flags_edge = checkpoint['candidate_flags_edge']
        self.proj_weights = checkpoint['proj_weights']