from torch import optim

optim_map = {
    name.lower(): val for name, val in optim.__dict__.items() 
    if isinstance(val, type) and val != optim.Optimizer
}

def get_optimizer(name: str, params, **kwargs):
    if name not in optim_map:
        raise NotImplementedError(f'\'{name}\' is not implemented. Supports ' + ', '.join(optim_map.keys()))

    return optim_map[name](params=params, **kwargs)
