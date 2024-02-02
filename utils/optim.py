import torch.optim as optim


def get_optimizer(params, lr, weight_decay):
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def get_lr_scheduler(optimizer, tinfo, milestones=None, gamma=None, start_epoch=-1, verbose=True):

    if tinfo['multi_step']:
        print('multi step!')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                   gamma=gamma, last_epoch=start_epoch,
                                                   verbose=verbose)
    elif tinfo['exponential_step']:
        print('exponential step!')
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92,
                                                     last_epoch=start_epoch,
                                                     verbose=verbose)
    elif tinfo['cosine_step']:
        print('Cosine step!')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120,
                                                         verbose=verbose)

    return scheduler
