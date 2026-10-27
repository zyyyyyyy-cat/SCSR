import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    scsr_params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if key.startswith('base.scsr_v41.'):
            scsr_params.append(value)
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if scsr_params:
        params.append({"params": scsr_params,
                       "lr": cfg.SOLVER.BASE_LR * cfg.SCSR_V41.SCSR_LR_MULTIPLIER,
                       "weight_decay": cfg.SCSR_V41.SCSR_WEIGHT_DECAY,
                       "name": 'SCSR-V4.1'})
    flat = [id(p) for group in params for p in group['params']]
    if len(flat) != len(set(flat)) or set(flat) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer must contain each trainable parameter exactly once')

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
