import torch

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr=0.0009, peak_lr=0.09):
        import torch.optim.lr_scheduler as lr_scheduler
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.peak_lr = peak_lr
        self.step_num = 0
        self.reduce_lr_on_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10)

    def get_lr(self):
        if self.step_num < self.warmup_steps:
            lr = self.base_lr + self.step_num * (self.peak_lr - self.base_lr) / self.warmup_steps
        else:
            lr = self.peak_lr
        return lr

    def step(self, loss=None):
        self.step_num += 1

        if self.step_num <= self.warmup_steps:
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if loss is not None:
                self.reduce_lr_on_plateau.step(loss)
