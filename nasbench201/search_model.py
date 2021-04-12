import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
from torch.autograd import Variable

class TinyNetwork(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args, affine=False, track_running_stats=True):
    super(TinyNetwork, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self._num_classes = num_classes
    self._criterion = criterion
    self._args = args
    self._affine = affine
    self._track_running_stats = track_running_stats
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.num_edge   = num_edge
    self.num_op     = len(search_space)
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    # self._arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    self._arch_parameters = Variable(1e-3*torch.randn(num_edge, len(search_space)).cuda(), requires_grad=True)

    ## optimizer
    arch_params = set(id(m) for m in self.arch_parameters())
    self._model_params = [m for m in self.parameters() if id(m) not in arch_params]

    self.optimizer = torch.optim.SGD(
        self._model_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
  
  def entropy_y_x(self, p_logit):
    p = F.softmax(p_logit, dim=1)
    return - torch.sum(p * F.log_softmax(p_logit, dim=1)) / p_logit.shape[0]

  def _loss(self, input, target, return_logits=False):
    logits = self(input)
    loss = self._criterion(logits, target)
    
    return (loss, logits) if return_logits else loss

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def arch_parameters(self):
    return [self._arch_parameters]

  def get_theta(self):
    return nn.functional.softmax(self._arch_parameters, dim=-1).cpu()

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self._arch_parameters[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs, weights=None):
    sim_nn = []

    weights = nn.functional.softmax(self._arch_parameters, dim=-1) if weights is None else weights
    
    if self.slim:
      weights[1].data.fill_(0)
      weights[3].data.fill_(0)
      weights[4].data.fill_(0)

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

  def _save_arch_parameters(self):
    self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]

  def project_arch(self):
    self._save_arch_parameters()
    for p in self.arch_parameters():
      m, n = p.size()
      maxIndexs = p.data.cpu().numpy().argmax(axis=1)
      p.data = self.proximal_step(p, maxIndexs)

  def proximal_step(self, var, maxIndexs=None):
    values = var.data.cpu().numpy()
    m, n = values.shape
    alphas = []
    for i in range(m):
      for j in range(n):
        if j == maxIndexs[i]:
          alphas.append(values[i][j].copy())
          values[i][j] = 1
        else:
          values[i][j] = 0
    return torch.Tensor(values).cuda()

  def restore_arch_parameters(self):
    for i, p in enumerate(self._arch_parameters):
      p.data.copy_(self._saved_arch_parameters[i])
    del self._saved_arch_parameters

  def new(self):
    model_new = TinyNetwork(self._C, self._layerN, self.max_nodes, self._num_classes, self._criterion,
                            self.op_names, self._args, self._affine, self._track_running_stats).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
      x.data.copy_(y.data)

    return model_new

  def step(self, input, target, args, shared=None, return_grad=False):
    Lt, logit_t = self._loss(input, target, return_logits=True)
    Lt.backward()
    nn.utils.clip_grad_norm_(self.get_weights(), args.grad_clip)
    self.optimizer.step()

    if return_grad:
      grad = torch.nn.utils.parameters_to_vector([p.grad for p in self.get_weights()])
      return logit_t, Lt, grad
    else:
      return logit_t, Lt

  def printing(self, logging):
    logging.info(self.get_theta())
  
  def set_arch_parameters(self, new_alphas):
    for alpha, new_alpha in zip(self.arch_parameters(), new_alphas):
        alpha.data.copy_(new_alpha.data)

  def save_arch_parameters(self):
    self._saved_arch_parameters = self._arch_parameters.clone()
  
  def restore_arch_parameters(self):
    self.set_arch_parameters(self._saved_arch_parameters)
    
  def reset_optimizer(self, lr, momentum, weight_decay):
    del self.optimizer
    self.optimizer = torch.optim.SGD(
      self.get_weights(),
      lr,
      momentum=momentum,
      weight_decay=weight_decay)