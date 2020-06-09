import torch
import torchvision.models as models

from configs.defaults import get_cfg
from slowfast.models.video_model_builder import ResNet, SlowFast, X3D


def build_model(name, opts):
    is_3d = True
    if name == 'resnet18':
        model = models.resnet18()
        is_3d = False
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        is_3d = False
    elif name == 'i3d':
        cfg = get_cfg()
        cfg.merge_from_file("configs/I3D.yaml")
        cfg.merge_from_list(opts)
        model = ResNet(cfg)
    elif name == 'x3d':
        cfg = get_cfg()
        cfg.merge_from_file("configs/X3D_M.yaml")
        cfg.merge_from_list(opts)
        model = X3D(cfg)
    else:
        raise NotImplementedError()
    
    print("Profile model {}\n".format(name))
    print("Model info: {}\n".format(model))
    device = torch.cuda.current_device()
    model = model.cuda(device=device)

    return model, is_3d
