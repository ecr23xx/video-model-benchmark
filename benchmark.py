import torch
import torch.nn as nn
from torchprof import Profile
from misc import log_model_info
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


def benchmark_run(model, name, pattern=None):
    log_model_info(model)
    device = torch.cuda.current_device()

    pattern = ["b"]
    paths = list(walk_modules(model, pattern=pattern))
    paths = None

    labels = torch.empty(4, dtype=torch.long).random_(400).to(device)
    loss_func = nn.CrossEntropyLoss(reduction='mean').to(device)

    with Profile(model, paths=paths, use_cuda=True) as prof:
        inputs = [torch.rand(4, 3, 16, 224, 224).to(device)]
        output = model(inputs)
    print("Forward pass")
    print(prof.display(show_events=False))
    print("")

    # TODO: backward each layer
    # print("Backward pass")
    # with BackwardProfile(model, paths=paths, use_cuda=True) as bprof:
    #     inputs = [torch.rand(4, 3, 16, 224, 224).to(device)]
    #     output = model(inputs)
    #     loss = loss_func(output, labels)
    #     loss.backward()
    # print(bprof.display(show_events=False))
    # print("")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(10):  # compute averages for 10 trials
            inputs = [torch.rand(4, 3, 16, 224, 224).to(device)]
            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace("./logs/{}.out".format(name))
