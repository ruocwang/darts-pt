import torch
import torch.nn as nn
from .search_cells import NAS201SearchCell as SearchCell
from .search_model import TinyNetwork as TinyNetwork


class TinyNetworkDarts(TinyNetwork):
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True):
    super(TinyNetworkDarts, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
          affine=affine, track_running_stats=track_running_stats)

    self.theta_map = lambda x: torch.softmax(x, dim=-1)
  
  def get_theta(self):
    return self.theta_map(self._arch_parameters).cpu()

  def forward(self, inputs):
    weights = self.theta_map(self._arch_parameters)
    feature = self.stem(inputs)

    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, weights)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return logits
