import torch
import torch.nn as nn
import torchvision.models as models

from configs.defaults import get_cfg
from slowfast.models.resnet_helper import ResStage
from slowfast.models.video_model_builder import ResNet, X3D
from slowfast.models.head_helper import ResNetBasicHead


class SimpleNet(nn.Module):
    def __init__(self, trans_func, group=False):
        super().__init__()

        width_per_group = 16
        dim_inner = width_per_group
        d1, d2, d3 = 2, 2, 2
        strides = [[1], [2], [2], [2]]

        if group:
            num_groups = dim_inner
        else:
            num_groups = 1

        self.s1 = ResStage(
            dim_in=[3],
            dim_out=[width_per_group],
            dim_inner=[dim_inner],
            temp_kernel_sizes=[[3]],
            stride=strides[0],
            num_blocks=[d1],
            num_groups=[num_groups],
            num_block_temp_kernel=[d1],
            nonlocal_inds=[[]],
            nonlocal_group=[1],
            nonlocal_pool=[1, 2, 2],
            instantiation="softmax",
            trans_func_name=trans_func,
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=nn.BatchNorm3d,
        )

        self.s2 = ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=[[3]],
            stride=strides[1],
            num_blocks=[d1],
            num_groups=[num_groups],
            num_block_temp_kernel=[d1],
            nonlocal_inds=[[]],
            nonlocal_group=[1],
            nonlocal_pool=[1, 2, 2],
            instantiation="softmax",
            trans_func_name=trans_func,
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=nn.BatchNorm3d,
        )

        self.head = ResNetBasicHead([width_per_group * 4], 400, pool_size=[16])

    def forward(self, x):
        out = self.head(self.s2(self.s1(x)))
        return out


def build_model(name, opts):
    if name == 'i3d':
        cfg = get_cfg()
        cfg.merge_from_file("configs/I3D.yaml")
        cfg.merge_from_list(opts)
        model = ResNet(cfg)
    elif name == 'x3d':
        cfg = get_cfg()
        cfg.merge_from_file("configs/X3D_M.yaml")
        cfg.merge_from_list(opts)
        model = X3D(cfg)
    elif name == 'x3d_group_bottleneck':
        model = SimpleNet('x3d_transform', group=True)
    elif name == 'x3d_bottleneck':
        model = SimpleNet('x3d_transform', group=False)
    elif name == 'group_bottleneck':
        model = SimpleNet('bottleneck_transform', group=True)
    elif name == 'bottleneck':
        model = SimpleNet('bottleneck_transform', group=False)
    else:
        raise NotImplementedError()

    print("Profile model {}\n".format(name))
    print("Model info: {}\n".format(model))
    device = torch.cuda.current_device()
    model = model.cuda(device=device)

    return model
