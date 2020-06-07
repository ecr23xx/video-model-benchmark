import torch
import torchprof
from misc import log_model_info


def benchmark_run(model, is_3d=False):
    log_model_info(model, is_3d)
    device = torch.cuda.current_device()
    if is_3d:
        inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
    else:
        inputs = torch.rand(8, 3, 224, 224).to(device)

    name = type(model).__name__
    if name == 'X3D':
        paths = [
            ("X3D", "s1"), ("X3D", "s2"), ("X3D", "s3"),
            ("X3D", "s4"), ("X3D", "s5"),
            ("X3D", "pathway0_conv5"), ("X3D", "head")]
    elif name == 'ResNet':
        paths = [
            ("ResNet", "s1"), ("ResNet", "s2"), ("ResNet", "s3"),
            ("ResNet", "s4"), ("ResNet", "s5"),
            ("ResNet", "head")]
    else:
        paths = None

    with torchprof.Profile(model, paths=paths, use_cuda=True) as prof:
        model(inputs)
    print(prof.display(show_events=False))
