import torch
from profile import Profile
from misc import log_model_info


def benchmark_run(model, is_3d=False):
    log_model_info(model, is_3d)
    device = torch.cuda.current_device()
    if is_3d:
        inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
    else:
        inputs = torch.rand(8, 3, 224, 224).to(device)

    name = type(model).__name__
    if is_3d:
        paths = [("b",), ("s1",), ("s2",), ("s3",),
                 ("s4",), ("s5",), ("head",)]
    else:
        paths = None

    with Profile(model, paths=paths, use_cuda=True) as prof:
        model(inputs)
    print(prof.display(show_events=False))
