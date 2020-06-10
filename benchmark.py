import torch
import torch.nn as nn
from time import time
from torchprof import Profile
from misc import log_model_info
import torch.autograd.profiler as tprofiler
from collections import namedtuple, defaultdict, OrderedDict

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def backward_tic(self, grad_inputs, grad_outputs):
    self.tic = time()


def flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(flatten_tree(st, depth=depth + 1))
    return flat


class BackwardProfile(object):
    """Layer by layer profiling of Pytorch models, using the Pytorch autograd profiler.
    """

    def __init__(self, model, enabled=True, use_cuda=False, paths=None):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._handles = []  # store the original forward functions
        self.traces = tuple(
            map(self._hook_trace, self.walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        self.traces = tuple(
            map(self._hook_trace, self.walk_modules(self._model)))
        for handle in self._handles:
            handle.remove()
        del self._handles
        self.exited = True

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        if leaf:
            handle = module.register_backward_hook(backward_tic)
            self._handles.append(handle)
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        if leaf:
            print(module.tic)
            # self.trace_profile_events[path] = module.tic

    def walk_modules(self, module, name="", path=()):
        """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
        if not name:
            name = module.__class__.__name__
        named_children = list(module.named_children())
        path = path + (name,)
        yield Trace(path, len(named_children) == 0, module)
        # recursively walk into all submodules
        for name, child_module in named_children:
            yield from self.walk_modules(child_module, name=name, path=path)

    def display(self):
        if not self.exited:
            return "<unfinished torchprof.profile>"
        
        tree = OrderedDict()

        for trace in self.traces:
            [path, leaf, module] = trace
            current_tree = tree
            for depth, name in enumerate(path, 1):
                if name not in current_tree:
                    current_tree[name] = OrderedDict()
                if depth == len(path) and leaf:
                    # tree measurements have key None, avoiding name conflict
                    if hasattr(module, 'tic'):
                        current_tree[name][None] = module.tic
                current_tree = current_tree[name]
        tree_lines = flatten_tree(tree)

        tic = None
        for idx in reversed(range(len(tree_lines))):
            depth, name, measures = tree_lines[idx]
            if isinstance(measures, float):
                if tic == None:
                    tic = measures
                    tree_lines[idx][2] = ""
                else:
                    tree_lines[idx][2] = tprofiler.format_time(((tree_lines[idx][2] - tic) * 1e6))
                    tic = measures
            else:
                tree_lines[idx][2] = ""

        # dt = ('|', '|-- ', '+-- ', ' ') # ascii
        dt = ("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " ")  # ascii-ex
        format_lines = []
        for idx, tree_line in enumerate(tree_lines):
            depth, name, measures = tree_line
            pre = ""
            next_depths = [pl[0] for pl in tree_lines[idx + 1 :]]
            current = True
            while depth:
                if current:
                    if depth in next_depths and next_depths[0] >= depth:
                        pre = dt[1]
                    else:
                        pre = dt[2]
                else:
                    if depth in next_depths:
                        pre = dt[0] + pre
                    else:
                        pre = dt[3] + pre
                depth -= 1
                current = False
            format_lines.append([pre + name, measures])

        # construct the table
        heading = ("Module", "Time Total")
        max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]
        # create the heading
        disp = "{:<{}s}".format(heading[0], max_lens[0]) + " | "
        disp += "{:>{}s}".format(heading[1], max_lens[1]) + "\n"
        disp += "-|-".join(["-" * mlen for mlen in max_lens]) + "\n"
        for line in format_lines:
            label, time_total = line
            disp += "{:<{}s}".format(label, max_lens[0]) + " | "
            disp += "{:>{}s}".format(time_total, max_lens[1]) + "\n"

        return disp


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
    print("Backward pass")
    with BackwardProfile(model, paths=paths, use_cuda=True) as bprof:
        inputs = [torch.rand(4, 3, 16, 224, 224).to(device)]
        output = model(inputs)
        loss = loss_func(output, labels)
        loss.backward()
    print(bprof.display())
    print("")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(10):  # compute averages for 10 trials
            inputs = [torch.rand(4, 3, 16, 224, 224).to(device)]
            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace("./logs/{}.out".format(name))
