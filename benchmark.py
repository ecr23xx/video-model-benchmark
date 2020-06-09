import torch
from torchprof import Profile
from misc import log_model_info
from collections import namedtuple, defaultdict, OrderedDict


def walk_modules(module, name="", path=(), pattern=[]):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    if name in pattern:
        yield path
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path, pattern=pattern)


def benchmark_run(model, is_3d=False):
    log_model_info(model, is_3d)
    device = torch.cuda.current_device()
    if is_3d:
        inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
    else:
        inputs = torch.rand(8, 3, 224, 224).to(device)

    if is_3d:
        pattern = ["b"]
        paths = list(walk_modules(model, pattern=pattern))
        paths = None
    else:
        pattern = []
        paths = None

    with Profile(model, paths=paths, use_cuda=True) as prof:
        model(inputs)
    print(prof.display(show_events=False))
