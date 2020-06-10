import torch
import torch.nn as nn
from misc import log_model_info
from profile import Profile, BackwardProfile
from collections import namedtuple, defaultdict, OrderedDict


def walk_modules(module, name="", path=(), pattern=[]):
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

    pattern = ["b"]
    paths = list(walk_modules(model, pattern=pattern))
    paths = None

    labels = torch.empty(8, dtype=torch.long).random_(400).to(device)
    loss_func = nn.CrossEntropyLoss(reduction='mean').to(device)

    with Profile(model, paths=paths, use_cuda=True) as prof:
        inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
        output = model(inputs)
        loss = loss_func(output, labels)
        loss.backward()
    print("Forward pass")
    print(prof.display(show_events=False))
    print("")

    print("Backward pass")
    with BackwardProfile(model, paths=paths, use_cuda=True) as bprof:
        inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
        output = model(inputs)
        loss = loss_func(output, labels)
        loss.backward()
    print(bprof.display(show_events=False))
    print("")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(10):
            inputs = [torch.rand(8, 3, 16, 224, 224).to(device)]
            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total"))
    print(prof.key_averages().table(sort_by="cuda_time_total"))
