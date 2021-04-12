import torch


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                        lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                        weight_decay=args.arch_weight_decay)

        self._init_arch_parameters = []
        for alpha in self.model.arch_parameters():
            alpha_init = torch.zeros_like(alpha)
            alpha_init.data.copy_(alpha)
            self._init_arch_parameters.append(alpha_init)

        #### mode
        if args.method in ['darts', 'darts-proj', 'sdarts', 'sdarts-proj']:
            self.method = 'fo' # first order update
        elif 'so' in args.method:
            print('ERROR: PLEASE USE architect.py for second order darts')
        elif args.method in ['blank', 'blank-proj']:
            self.method = 'blank'
        else:
            print('ERROR: WRONG ARCH UPDATE METHOD', args.method); exit(0)

    def reset_arch_parameters(self):
        for alpha, alpha_init in zip(self.model.arch_parameters(), self._init_arch_parameters):
            alpha.data.copy_(alpha_init.data)

    def step(self, input_train, target_train, input_valid, target_valid, *args, **kwargs):
        if self.method == 'fo':
            shared = self._step_fo(input_train, target_train, input_valid, target_valid)
        elif self.method == 'so':
            raise NotImplementedError
        elif self.method == 'blank': ## do not update alpha
            shared = None

        return shared

    #### first order
    def _step_fo(self, input_train, target_train, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        self.optimizer.step()
        return None

    #### darts 2nd order
    def _step_darts_so(self, input_train, target_train, input_valid, target_valid, eta, model_optimizer):
        raise NotImplementedError