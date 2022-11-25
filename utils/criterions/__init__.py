import torch.nn.functional as F

loss_map = {
    'ce': F.cross_entropy,
    'cross_entropy': F.cross_entropy,
}

def get_loss(name: str):
    if name not in loss_map:
        raise NotImplementedError(f'\'{name}\' is not implemented. Supports ' + ', '.join(loss_map.keys()))
    
    return loss_map[name]
