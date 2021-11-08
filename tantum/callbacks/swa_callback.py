from torch.optim import swa_utils


## https://githubmemory.com/repo/skorch-dev/skorch/issues/700


class StochasticWeightAveraging(Callback):
    def __init__(
            self,
            swa_utils,
            swa_start=10,
            verbose=0,
            sink=print,
            **kwargs  # additional arguments to swa_utils.SWALR
    ):
        self.swa_utils = swa_utils
        self.swa_start = swa_start
        self.verbose = verbose
        self.sink = sink
        vars(self).update(kwargs)

    @property
    def kwargs(self):
        # These are the parameters that are passed to SWALR.
        # Parameters that don't belong there must be excluded.
        excluded = {'swa_utils', 'swa_start', 'verbose', 'sink'}
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        self.optimizer_swa_ = self.swa_utils.SWALR(net.optimizer_, **self.kwargs)
        if not hasattr(net, 'module_swa_'):
            net.module_swa_ = self.swa_utils.AveragedModel(net.module_)
            
    def on_epoch_begin(self, net, **kwargs):
        if self.verbose and len(net.history) == self.swa_start + 1:
            self.sink("Using SWA to update parameters")

    def on_epoch_end(self, net, **kwargs):
        if len(net.history) >= self.swa_start + 1:
            net.module_swa_.update_parameters(net.module_)
            self.optimizer_swa_.step()

    def on_train_end(self, net, X, y=None, **kwargs):
        if self.verbose:
            self.sink("Using training data to update batch norm statistics of the SWA model")

        loader = net.get_iterator(net.get_dataset(X, y))
        self.swa_utils.update_bn(loader, net.module_swa_, device = net.device)