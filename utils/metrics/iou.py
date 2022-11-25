from torch import nn

def intersection_over_union(output, target, class_id: int=-1, eps=1.0E-6):
    if class_id == -1:  # MIoU
        batch_size = output.size()[0]
    
        intersection = ((output.int() > 0) & (target.int() > 0) & (output.int() == target.int())).float()
        intersection = intersection.view(batch_size, -1).sum(1)
    
        union = ((output.int() > 0) | (target.int() > 0)).float()
        union = union.view(batch_size, -1).sum(1)
    
        iou = (intersection + eps) / (union + eps)
    
        return iou.mean()
    else:  # class IoU
        batch_size = output.size()[0]
            
        intersection = ((output.int() == class_id) & (target.int() == class_id) & (output.int() == target.int())).float()
        intersection = intersection.view(batch_size, -1).sum(1)

        union = ((output.int() == class_id) | (target.int() == class_id)).float()
        union = union.view(batch_size, -1).sum(1)

        iou = (intersection + eps) / (union + eps)
            
        return iou.mean()


class IoU(nn.Module):
    def __init__(self, class_id: int=-1, eps: float=1.0E-6):
        super(IoU, self).__init__()
        
        self.class_id = class_id
        self.eps = eps
        
    def forward(self, output, target):
        return intersection_over_union(output, target, self.class_id, self.eps)
