from __future__ import division, absolute_import
from torch.nn import functional as F
import torch.nn as nn
import warnings
import pickle
from functools import partial
import torch
import os.path as osp
from collections import OrderedDict


class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution (bias discarded) + batch normalization + relu6.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
            to output channels (default: 1).
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_c, out_c, k, stride=s, padding=p, bias=False, groups=g
        )

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):

        super(Bottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)

        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        m = self.conv1(x)

        m = self.dwconv2(m)

        m = self.conv3(m)

        if self.use_residual:
            return x + m

        else:
            return m


class MobileNetV2_deep(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """
    """
#################### Original Version of MobileNetV2 ######################

in_height, in_width, e_in_ch, e_out_ch, stride, kernel_size, groups,

    Suppose the size of input is 224 x 224 x 3

                Input Size          Output Size         # Bottleneck (17)   # Conv Layers (53)
        conv1:  224 x 224 x 3       112 x 112 x 32      0                   1
        conv2:  112 x 112 x 32      112 x 112 x 16      1                   3

Stage1  conv3:  112 x 112 x 16      56  x 56  x 24      2                   6

Stage2  conv4:  56  x 56  x 24      28  x 28  x 32      3                   9

Stage3  conv5:  28  x 28  x 32      14  x 14  x 64      4                   12
        conv6:  14  x 14  x 64      14  x 14  x 96      3                   9

Stage4  conv7:  14  x 14  x 96       7  x  7  x 160     3                   9
        conv8:   7  x  7  x 160      7  x  7  x 320     1                   3

        conv9:   7  x  7  x 320      7  x  7  x 1280    0                   1

#################### Deeper Version of MobileNetV2 ######################

    The number of bottleneck blocks is defined in structure[].
                                    # Bottleneck at each stage              # Bottleneck
                                                                                 __  __  ______  _____                 
    MobileNetV2-53(Original)        [2,  3,  7,  4]                         [1,  2,  3,  4,  3,  3,  1]                                                                                     
    MobileNetV2-107                 [3,  4, 23,  4]                         [1,  3,  4, 12, 11,  3,  1]     
    MobileNetV2-161                 [3,  8, 37,  4]                         [1,  3,  8, 19, 18,  3,  1]     
    MobileNetV2-200                 [3, 21, 37,  4]                         [1,  3, 21, 19, 18,  3,  1]
    MobileNetV2-299                 [3, 31, 60,  4]                         [1,  3, 31, 30, 30,  3,  1]

    ResNet-50                       [3,  4,  6,  3]
    ResNet-101                      [3,  4, 23,  3]
    ResNet-152                      [3,  8, 36,  3]
    ResNet-200                      [3, 24, 36,  3]

#################### MobileNetV2 with last_stride = 1 ######################

    Modify the stride of Stage-4 from 2 to 1

                Input Size          Output Size         # Bottleneck (17)   # Conv Layers (53)
        conv1:  224 x 224 x 3       112 x 112 x 32      0                   1
        conv2:  112 x 112 x 32      112 x 112 x 16      1                   3

Stage1  conv3:  112 x 112 x 16      56  x 56  x 24      2                   6

Stage2  conv4:  56  x 56  x 24      28  x 28  x 32      3                   9

Stage3  conv5:  28  x 28  x 32      14  x 14  x 64      4                   12
        conv6:  14  x 14  x 64      14  x 14  x 96      3                   9

Stage4  conv7:  14  x 14  x 96      14  x 14  x 160     3                   9
        conv8:  14  x 14  x 160     14  x 14  x 320     1                   3

        conv9:  14  x 14  x 320     14  x 14  x 1280    0                   1
    """

    def __init__(
            self,
            num_classes,
            width_mult=1.0,
            # loss='softmax',
            structure=[1, 2, 3, 4, 3, 3, 1],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
    ):
        super(MobileNetV2_deep, self).__init__()
        # self.loss = loss
        self.in_channels = int(32 * width_mult)
        # self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280
        self.feature_dim = int(1280 * width_mult)
        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), structure[0], 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), structure[1], 2)
        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), structure[2], 2)
        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), structure[3], 2)
        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), structure[4], 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), structure[5], last_stride)
        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), structure[6], 1)
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, t, c, n, s):
        # t: expansion factor
        # c: output channels
        # n: number of blocks
        # s: stride for first layer
        layers = []
        layers.append(block(self.in_channels, c, t, s))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        return f
        # v = self.global_avgpool(f)
        # v = v.view(v.size(0), -1)
        #
        # if self.fc is not None:
        #     v = self.fc(v)
        #
        # if not self.training:
        #     return v

        # y = self.classifier(v)
        #
        # if self.loss == 'softmax':
        #     return y
        # elif self.loss == 'triplet':
        #     return y, v
        # else:
        #     raise KeyError("Unsupported loss: {}".format(self.loss))


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.
    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 n_feature,
                 n_hidden,
                 spatial_dims=[2, 3],
                 active_fn=None):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        self.active_fn = active_fn()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x

    def __repr__(self):
        return '{}({}, {}, spatial_dims={}, active_fn={})'.format(
            self._get_name(), self.n_feature, self.n_hidden, self.spatial_dims,
            self.active_fn)


def mobilenetv2_53(num_classes, last_stride=2, width_mult=1.0, **kwargs):
    model = MobileNetV2_deep(
        num_classes,
        width_mult=width_mult,
        structure=[1, 2, 3, 4, 3, 3, 1],
        last_stride=last_stride,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    model._init_params()
    warnings.warn("Training mobilenetv2_53 from scratch.")
    return model


def mobilenetv2_107(num_classes, last_stride=2, width_mult=1, **kwargs):
    model = MobileNetV2_deep(
        num_classes,
        # loss=loss,
        width_mult=width_mult,
        structure=[1, 2 + 1, 3 + 1, 4 + 8, 3 + 8, 3, 1],
        last_stride=last_stride,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    model._init_params()
    warnings.warn("Training mobilenetv2_107 from scratch.")
    return model


def mobilenetv2_161(num_classes, last_stride=2, width_mult=1, **kwargs):
    model = MobileNetV2_deep(
        num_classes,
        # loss=loss,
        width_mult=width_mult,
        structure=[1, 2 + 1, 3 + 1 + 4, 4 + 8 + 7, 3 + 8 + 7, 3, 1],
        last_stride=last_stride,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    model._init_params()
    warnings.warn("Training mobilenetv2_161 from scratch.")
    return model


def mobilenetv2_200(num_classes, last_stride=2, width_mult=1, **kwargs):
    model = MobileNetV2_deep(
        num_classes,
        # loss=loss,
        width_mult=width_mult,
        structure=[1, 3, 21, 19, 18, 3, 1],
        last_stride=last_stride,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    model._init_params()
    warnings.warn("Training mobilenetv2_161 from scratch.")
    return model


def mobilenetv2_300(num_classes, last_stride=2, width_mult=1, **kwargs):
    model = MobileNetV2_deep(
        num_classes,
        # loss=loss,
        width_mult=width_mult,
        structure=[1, 3, 31, 30, 30, 3, 1],
        last_stride=last_stride,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    model._init_params()
    warnings.warn("Training mobilenetv2_300 from scratch.")
    return model



