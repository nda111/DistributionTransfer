from collections import OrderedDict

import torch
from torch import nn

import segmentation_models_pytorch as smp

from models.gscnn.network.gscnn import GSCNN


def __kaiming_normal(module: nn.Module):
    for p in module.parameters():
        if p.ndim > 1:
            nn.init.kaiming_normal_(p)


def __get_gscnn(**kwargs):
    info = torch.load(kwargs['params'])
    renamed_state_dict = OrderedDict(**{
        name[7:]: tensor for name, tensor in info['state_dict'].items()
    })

    net = GSCNN(num_classes=19, trunk='resnet101')
    net.load_state_dict(renamed_state_dict, strict=True)
    return net


def __get_unet(**kwargs):
    net = smp.Unet(
        classes=19,
        in_channels=3,
        encoder_weights=None,
        activation='sigmoid',
        **kwargs
    )
    __kaiming_normal(net)
    
    return net


def __get_unet_pp(**kwargs):
    net = smp.UnetPlusPlus(
        classes=19,
        in_channels=3,
        encoder_weights=None,
        activation='sigmoid',
        **kwargs
    )
    __kaiming_normal(net)
    
    return net
    
    
model_map = {
    'gscnn': __get_gscnn,
    'unet': __get_unet,
    'unet++': __get_unet_pp,
}


def get_model(name: str, device: str=None, **kwargs):
    if name not in model_map:
        raise NotImplementedError(f'\'{name}\' is not implemented. Supports ' + ', '.join(model_map.keys()))
    
    if device is None:
        device = 'cpu'

    return model_map[name](**kwargs).to(device)
    