import torch.nn.functional as F

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from sota.cnn.model_search_darts_proj import DartsNetworkProj


class SDartsNetworkProj(DartsNetworkProj):
    def __init__(self, C, num_classes, layers, criterion, primitives, args,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(SDartsNetworkProj, self).__init__(C, num_classes, layers, criterion, primitives, args,
              steps=steps, multiplier=multiplier, stem_multiplier=stem_multiplier, drop_path_prob=drop_path_prob)

        self.softmaxed = False

    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]
  
    def softmax_arch_parameters(self):
        self._save_arch_parameters()
        for p, cell_type in zip(self._arch_parameters, self.candidate_flags.keys()):
            p.data.copy_(self.get_projected_weights(cell_type))
        self.softmaxed = True # after self.get_projected_weights

    def restore_arch_parameters(self):
        for i, p in enumerate(self._arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters
        self.softmaxed = False

    def get_softmax(self):
        if self.softmaxed:
            weights_normal = self.alphas_normal
            weights_reduce = self.alphas_reduce
        else:
            weights_normal = F.softmax(self.alphas_normal, dim=-1)
            weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        
        return {'normal':weights_normal, 'reduce':weights_reduce}

    def arch_parameters(self):
        return self._arch_parameters