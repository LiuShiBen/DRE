import torch


def make_optimizer(args, model):
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = 1e-4
        '''if "bias" in key:
            lr = args.lr * 2
            weight_decay = 1e-4'''
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, momentum=0.9)

    return optimizer