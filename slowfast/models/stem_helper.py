#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t 3D stem helper."""

import torch.nn as nn


def get_stem_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "resnet_stem": ResNetBasicStem,
        "x3d_stem": X3DBasicStem,
    }
    assert (
        name in trans_funcs.keys()
    ), "Stem function '{}' not supported".format(name)
    return trans_funcs[name]


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        stem_func_name,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            stem_func_name (str):
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, stem_func_name, norm_module)

    def _construct_stem(self, dim_in, dim_out, stem_func_name, norm_module):
        stem_func = get_stem_func(stem_func_name)
        for pathway in range(len(dim_in)):
            stem = stem_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        is_pool=True,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            is_pool (bool): pooling or not
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.is_pool = is_pool

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        if self.is_pool:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.is_pool:
            x = self.pool_layer(x)
        return x


class X3DBasicStem(nn.Module):
    """
    X3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        # 1x3x3
        self.conv1 = nn.Conv3d(
            dim_in,
            dim_out,
            [1] + self.kernel[1:],
            stride=self.stride,
            padding=[0] + self.padding[1:],
            bias=False,
        )
        self.bn1 = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )

        # 3x1x1
        self.conv2 = nn.Conv3d(
            dim_out,
            dim_out,
            [self.kernel[0]] + [1, 1],
            stride=self.stride,
            padding=[self.padding[0]] + [0, 0],
            groups=dim_out,
            bias=False,
        )
        self.bn2 = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )

        self.relu = nn.ReLU(self.inplace_relu)

        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x
