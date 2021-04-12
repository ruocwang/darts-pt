import torch.nn.functional as F

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
from sota.cnn.model_search import Network


class SDartsNetwork(Network):
    def __init__(self, C, num_classes, layers, criterion, primitives, args,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(SDartsNetwork, self).__init__(C, num_classes, layers, criterion, primitives, args,
                                               steps, multiplier, stem_multiplier, drop_path_prob)

        self.softmaxed = False

    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]
  
    def softmax_arch_parameters(self):
        self.softmaxed = True
        self._save_arch_parameters()
        for p in self._arch_parameters:
            p.data.copy_(F.softmax(p, dim=-1))

    def restore_arch_parameters(self):
        self.softmaxed = False
        for i, p in enumerate(self._arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters

    def get_softmax(self):
        if self.softmaxed:
            weights_normal = self.alphas_normal
            weights_reduce = self.alphas_reduce
        else:
            weights_normal = F.softmax(self.alphas_normal, dim=-1)
            weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        
        return {'normal':weights_normal, 'reduce':weights_reduce}
