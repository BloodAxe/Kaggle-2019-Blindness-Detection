from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules.activations import ACT_HARD_SWISH
from pytorch_toolbelt.modules.agn import AGN
from pytorch_toolbelt.modules.backbone.efficient_net import round_filters, round_repeats, drop_connect
from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_


class EfficientNetBlockArgs:
    def __init__(self, input_filters, output_filters, expand_ratio,
                 r=1,
                 k=3,
                 s=1,
                 se_reduction=4,
                 dropout=0.0,
                 id_skip=True,
                 **kwargs):
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.num_repeat = r
        self.se_reduction = se_reduction
        self.dropout = dropout
        self.kernel_size = k
        self.stride = s
        self.width_coefficient = 1.0
        self.depth_coefficient = 1.0
        self.depth_divisor = 8
        self.min_filters = None
        self.id_skip = id_skip

    def scale(self,
              width_coefficient: float,
              depth_coefficient: float,
              depth_divisor: float = 8.,
              min_filters: int = None):
        copy = deepcopy(self)
        copy.input_filters = round_filters(self.input_filters,
                                           width_coefficient, depth_divisor,
                                           min_filters)
        copy.output_filters = round_filters(self.output_filters,
                                            width_coefficient, depth_divisor,
                                            min_filters)
        copy.num_repeat = round_repeats(self.num_repeat, depth_coefficient)
        copy.width_coefficient = width_coefficient
        copy.depth_coefficient = depth_coefficient
        copy.depth_divisor = depth_divisor
        copy.min_filters = min_filters
        return copy


class SCSE(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module

        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super().__init__()
        assert reduction or squeeze_channels, "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"

        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_normal_(self.squeeze.weight, nonlinearity='relu')
        kaiming_normal_(self.expand.weight, nonlinearity='sigmoid')

    def forward(self, x: torch.Tensor):
        module_input = x
        x = self.avg_pool(x)
        x = self.squeeze(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        # print(x.data.mean(), x.data.std(), module_input.data.mean(), module_input.data.std())
        return module_input * x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args: EfficientNetBlockArgs,
                 abn_block: ABN,
                 abn_params):
        super().__init__()

        self._block_args = deepcopy(block_args)
        self.has_se = (block_args.se_reduction is not None)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.expand_ratio = block_args.expand_ratio

        # Expansion phase
        inp = block_args.input_filters  # number of input channels
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels

        if block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=inp,
                                         out_channels=oup,
                                         kernel_size=1,
                                         bias=False)
            kaiming_normal_(self.expand_conv.weight, nonlinearity=abn_params['activation'])
            self.abn0 = abn_block(oup, **abn_params)

        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=block_args.kernel_size,
            padding=block_args.kernel_size//2,
            stride=block_args.stride,
            bias=False)
        kaiming_normal_(self.depthwise_conv.weight,
                        nonlinearity=abn_params['activation'])

        self.abn1 = abn_block(oup, **abn_params)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            self.se_block = SCSE(oup,
                                 squeeze_channels=max(1, inp // block_args.se_reduction))

        # Output phase
        final_oup = self._block_args.output_filters
        self.project_conv = nn.Conv2d(in_channels=oup,
                                      out_channels=final_oup,
                                      kernel_size=1, bias=False)
        kaiming_normal_(self.project_conv.weight,
                        nonlinearity=abn_params['activation'])

        self.abn2 = abn_block(final_oup, **abn_params)

        self.input_filters = block_args.input_filters
        self.output_filters = block_args.output_filters

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self.expand_ratio != 1:
            # Expansion and Depthwise Convolution
            x = self.abn0(self.expand_conv(inputs))

        x = self.abn1(self.depthwise_conv(x))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se_block(x)

        x = self.abn2(self.project_conv(x))

        # Skip connection and drop connect
        if self.id_skip \
                and self._block_args.stride == 1 \
                and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x


def get_default_efficientnet_params(dropout=0.2, **kwargs):
    return [
        EfficientNetBlockArgs(r=1, k=3, s=1, expand_ratio=1, input_filters=32,
                              output_filters=16, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=2, k=3, s=2, expand_ratio=6, input_filters=16,
                              output_filters=24, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=2, k=5, s=2, expand_ratio=6, input_filters=24,
                              output_filters=40, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=3, k=3, s=2, expand_ratio=6, input_filters=40,
                              output_filters=80, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=3, k=5, s=1, expand_ratio=6, input_filters=80,
                              output_filters=112, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=4, k=5, s=2, expand_ratio=6, input_filters=112,
                              output_filters=192, se_reduction=4,
                              dropout=dropout, **kwargs),
        EfficientNetBlockArgs(r=1, k=3, s=1, expand_ratio=6, input_filters=192,
                              output_filters=320, se_reduction=4,
                              dropout=dropout, **kwargs),
    ]


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self,
                 blocks_args: List[EfficientNetBlockArgs],
                 num_classes: int,
                 abn_block=ABN,
                 abn_params=None,
                 in_channels=3,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'

        if abn_params is None:
            assert issubclass(abn_block, ABN)
            abn_params = {
                'activation': 'swish',
                'momentum': 0.01,
                'eps': 1e-3
            }

        first_block_args = blocks_args[0]
        last_block_args = blocks_args[-1]

        # Stem
        self.stem = nn.Sequential(OrderedDict([
            ("conv",
             nn.Conv2d(in_channels, first_block_args.input_filters,
                       kernel_size=3,
                       stride=2,
                       bias=False)),
            ("abn", abn_block(first_block_args.input_filters, **abn_params))
        ]))

        kaiming_normal_(self.stem.conv.weight, nonlinearity=abn_params['activation'])

        # Build blocks
        for i, block_args in enumerate(blocks_args):
            module = []

            # The first block needs to take care of stride and filter size increase.
            module.append(MBConvBlock(block_args, abn_block, abn_params))

            if block_args.num_repeat > 1:
                block_args = deepcopy(block_args)
                block_args.stride = 1
                block_args.input_filters = block_args.output_filters

            for _ in range(block_args.num_repeat - 1):
                module.append(MBConvBlock(block_args, abn_block, abn_params))

            self.add_module(f'block{i}', nn.Sequential(*module))

        # Head
        out_channels = round_filters(1280,
                                     last_block_args.width_coefficient,
                                     last_block_args.depth_divisor,
                                     last_block_args.min_filters)

        self.conv_head = nn.Conv2d(last_block_args.output_filters,
                                   out_channels, kernel_size=1, bias=False)
        self.abn_head = abn_block(out_channels, **abn_params)

        # Final linear layer
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, inputs):
        # Stem
        x = self.stem(inputs)
        print('stem', x.mean(), x.std(), x)

        # Blocks
        x = self.block0(x)

        print('block0', x.mean(), x.std(), x)
        print(x.size())
        x = self.block1(x)
        print(x.size())
        x = self.block2(x)
        print(x.size())
        x = self.block3(x)
        print(x.size())
        x = self.block4(x)
        print(x.size())
        x = self.block5(x)
        print(x.size())
        x = self.block6(x)
        print(x.size())

        # Head
        x = self.abn_head(self.conv_head(x))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def efficient_net_b0(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.2, **kwargs)
    params = [p.scale(width_coefficient=1.0, depth_coefficient=1.0) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b1(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.2)
    params = [p.scale(width_coefficient=1.0, depth_coefficient=1.1) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b2(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.3)
    params = [p.scale(width_coefficient=1.1, depth_coefficient=1.2) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b3(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.3)
    params = [p.scale(width_coefficient=1.2, depth_coefficient=1.4) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b4(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.3)
    params = [p.scale(width_coefficient=1.4, depth_coefficient=1.8) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b5(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.5)
    params = [p.scale(width_coefficient=1.6, depth_coefficient=2.2) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b6(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.5)
    params = [p.scale(width_coefficient=1.8, depth_coefficient=2.6) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def efficient_net_b7(num_classes: int, **kwargs):
    params = get_default_efficientnet_params(dropout=0.5)
    params = [p.scale(width_coefficient=2.0, depth_coefficient=3.1) for p in
              params]
    return EfficientNet(params, num_classes=num_classes, **kwargs)


def test_efficient_net():
    from pytorch_toolbelt.utils.torch_utils import count_parameters
    num_classes = 1001

    x = torch.randn((1, 3, 600, 600)).cuda()

    for model_fn in [efficient_net_b0,
                     # efficient_net_b1,
                     # efficient_net_b2,
                     # efficient_net_b3,
                     # efficient_net_b4,
                     # efficient_net_b5,
                     # efficient_net_b6,
                     # efficient_net_b7
                     ]:
        print('=======', model_fn.__name__, '=======')
        model = model_fn(num_classes).eval().cuda()
        print(count_parameters(model))
        print(model)
        print()
        print()

        output = model(x)
        print(output.size())


def test_efficient_net_group_norm():
    from pytorch_toolbelt.utils.torch_utils import count_parameters
    num_classes = 1001

    x = torch.randn((1, 3, 600, 600)).cuda()

    for model_fn in [efficient_net_b0,
                     efficient_net_b1,
                     efficient_net_b2,
                     efficient_net_b3,
                     efficient_net_b4,
                     efficient_net_b5,
                     efficient_net_b6,
                     efficient_net_b7]:
        print('=======', model_fn.__name__, '=======')
        agn_params = {
            'num_groups': 8,
            'activation': ACT_HARD_SWISH
        }
        model = model_fn(num_classes, abn_block=AGN,
                         abn_params=agn_params).eval().cuda()
        print(count_parameters(model))
        # print(model)
        print()
        print()

        output = model(x)
        print(output.size())
        torch.cuda.empty_cache()
