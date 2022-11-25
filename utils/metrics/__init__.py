from utils.metrics.iou import intersection_over_union, IoU

metric_map = {
    'iou': intersection_over_union,
    'intersection_over_union': intersection_over_union,
}

def get_metrics(name: str):
    if name not in metric_map:
        raise NotImplementedError(f'\'{name}\' is not implemented. Supports ' + ', '.join(metric_map.keys()))
    
    return metric_map[name]
