from torch.optim.lr_scheduler import ReduceLROnPlateau

class LFADS_Scheduler(ReduceLROnPlateau):
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        
        super(LFADS_Scheduler, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                                              verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                                              cooldown=cooldown, min_lr=min_lr, eps=eps)
        
        
    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.best = self.mode_worse