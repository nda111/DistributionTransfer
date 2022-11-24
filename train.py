import torch

from gscnn import GSCNN

# info = torch.load('./gscnn/best_cityscapes_checkpoint.pth')
# print(info.keys())
# print(info['state_dict']['module.final_seg.6.weight'].shape)
# exit()

net = GSCNN(num_classes=19, trunk='resnet101')
net.load_state_dict(torch.load('./gscnn/best_cityscapes_checkpoint.pth')['state_dict'], strict=True)
print(net)
