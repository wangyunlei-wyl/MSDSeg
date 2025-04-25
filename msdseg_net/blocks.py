import math

from torch import nn
import torch
from torch.nn import functional as F


def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)

class depthwise_separable_conv(nn.Module):
    def init(self, in_channels, out_channels):
        super(depthwise_separable_conv, self).init()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm2d(out_channels)
        if apply_act:
            self.act = activation()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


# class Shortcut(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(Shortcut, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False
class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.conv1 = ConvBnAct(in_channels=inplanes,out_channels=interplanes,kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out

class DilatedConv(nn.Module):
    def __init__(self, w, dilations, group_width, stride, bias):
        super().__init__()
        num_splits = len(dilations)
        assert (w % num_splits == 0)
        temp = w // num_splits
        assert (temp % group_width == 0)
        groups = temp // group_width
        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(temp, temp, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=groups))
        self.convs = nn.ModuleList(convs)
        self.num_splits = num_splits

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)


class ConvBnActConv(nn.Module):
    def __init__(self, w, stride, dilation, groups, bias):
        super().__init__()
        self.conv = ConvBnAct(w, w, 3, stride, dilation, dilation, groups)
        self.project = nn.Conv2d(w, w, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride, attention="se"):
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DilatedConv(out_channels, dilations, group_width=group_width, stride=stride, bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        if attention == "se":
            self.se = SEModule(out_channels, in_channels // 4)
        elif attention == "se2":
            self.se = SEModule(out_channels, out_channels // 4)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class Exp2_LRASPP(nn.Module):
    # LRASPP
    def __init__(self, num_classes, inter_channels=128):
        super().__init__()
        channels = {"8": 128, "16": 128}
        # channels8, channels16 = channels["8"], channels["16"]
        self.cbr = ConvBnAct(channels["16"], inter_channels, 1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels["16"], inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels["8"], num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["16"], x["32"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x = self.low_classifier(x8) + self.high_classifier(x)
        return x


class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channels = {"8": 64, "16": 128, "32": 256}
        self.head32 = ConvBnAct(channels['32'], 128, 1)
        self.head16 = ConvBnAct(channels['16'], 128, 1)
        self.head8 = ConvBnAct(channels['8'], 8, 1)
        self.conv16 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv8 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x8, x16, x32 = x["8"], x["16"], x["32"]
        x32 = self.head32(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x16 + x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = torch.cat((x16, x8), dim=1)
        x8 = self.conv8(x8)
        x8 = self.classifier(x8)
        return x8

# class CBA(nn.Module):
#     def __init__(self, in_ch, out_ch, ks=1, s=1, pad=0, dil=1, g=1,
#                  bias=False, act=True):
#         super(CBA, self).__init__()
#
#         self.name = "CBA"
#         self.ks = ks
#         self.s = s
#         self.dil = dil
#
#         # Get the number of group channels
#         self.g = get_groups(in_ch, out_ch)
#
#         self.conv = nn.Conv2d(in_ch, out_ch, ks, s, pad, dil, g, bias)
#         self.bn = nn.BatchNorm2d(out_ch)
#
#         if act:
#             self.act = REU()
#         else:
#             self.act = nn.Identity()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#
#         return x


class CS(nn.Module):
    def __init__(self, g):
        super(CS, self).__init__()

        self.name = "cs"
        self.g = g

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.g

        # Reshape into (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.g, channels_per_group, height, width)

        # Transpose (swap) dimensions 1 and 2
        x = x.transpose(1, 2).contiguous()

        # Flatten back into (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        # Return the shuffled tensor
        return x


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.name = "ECA"

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma

        # Calculate kernel size for 1D convolution
        kernel_size = int(abs((math.log2(channels) / gamma) + b / gamma))
        self.ks = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.ks,
            padding=(self.ks - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # 1D convolution to model channel-wise dependencies
        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Sigmoid activation for channel-wise attention weights
        y = self.sigmoid(y)

        # Multi-scale information fusion
        out = x * y.expand_as(x)

        return out


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.name = "SAM"

        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Channel-wise attention
        channel_out = self.conv1(x)
        channel_attention = self.sigmoid(channel_out)

        # Spatial-wise attention
        avg_spatial_attention = self.avg_pool(channel_attention)
        max_spatial_attention = self.max_pool(channel_attention)
        spatial_attention = avg_spatial_attention + max_spatial_attention

        # Element-wise multiplication
        x = x * channel_attention * spatial_attention

        return x


class CA(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8):
        super(CA, self).__init__()
        self.name = "CA"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes,
                             in_planes // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio,
                             in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.name = "SA"
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.name = "CBAM"
        self.channel_gate = CA(in_planes, reduction_ratio)
        self.spatial_gate = SA(kernel_size)

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


# from CBAM import CBAM

class Bag(nn.Module):
    def __init__(self, p_channels, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(p_channels, in_channels,
                      kernel_size=1, padding=0, bias=False),
            BatchNorm(in_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.atten = CA(in_channels)
        # self.atten = SA()

    def forward(self, p, i):
        # print(p.shape,i.shape)
        p = self.conv1(p)
        edge_att = self.atten(p + i)
        return self.conv(edge_att * p + (1 - edge_att) * i)
        # return edge_att * p + (1 - edge_att) * i



class decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channels = {"8": 64, "16": 128, "32": 256}
        self.head32 = ConvBnAct(channels['32'], 128, 1)
        self.head16 = ConvBnAct(channels['16'], 128, 1)
        self.head8 = ConvBnAct(channels['8'], 8, 1)
        self.conv16 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv8 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x8, x16, x32 = x["8"], x["16"], x["32"]
        x32 = self.head32(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x16 + x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = torch.cat((x16, x8), dim=1)
        x8 = self.conv8(x8)
        x8 = self.classifier(x8)
        return x8

# class Exp2_Decoder26(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         channels = {"4": 16, "8": 48, "16": 128, "32": 384}
#         self.head32 = ConvBnAct(channels["32"], 128, 1)
#         self.head16 = ConvBnAct(channels["16"], 128, 1)
#         self.head8 = ConvBnAct(channels["8"], 128, 1)
#         self.head4 = ConvBnAct(channels["4"], 8, 1)
#         self.conv16 = ConvBnAct(128, 128, 3, 1, 1)
#         self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
#         self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
#         self.classifier = nn.Conv2d(64, num_classes, 1)
#
#     def forward(self, x):
#         x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
#         x32 = self.head32(x32)
#         x16 = self.head16(x16)
#         x8 = self.head8(x8)
#         x4 = self.head4(x4)
#         x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
#         x16 = x16 + x32
#         x16 = self.conv16(x16)
#         x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
#         x8 = x8 + x16
#         x8 = self.conv8(x8)
#         x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
#         x4 = torch.cat((x8, x4), dim=1)
#         x4 = self.conv4(x4)
#         x4 = self.classifier(x4)
#         return x4

def generate_stage(num, block_fun):
    blocks = []
    for _ in range(num):
        blocks.append(block_fun())
    return blocks


def generate_stage2(ds, block_fun):
    blocks = []
    for d in ds:
        blocks.append(block_fun(d))
    return blocks



class CS(nn.Module):
    def __init__(self, g):
        super(CS, self).__init__()

        self.name = "cs"
        self.g = g

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.g

        # Reshape into (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.g, channels_per_group, height, width)

        # Transpose (swap) dimensions 1 and 2
        x = x.transpose(1, 2).contiguous()

        # Flatten back into (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        # Return the shuffled tensor
        return x

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        if self.use_relu:
            feat = self.relu(feat)
        else:
            feat = feat
        return feat


# class StemBlock(nn.Module):
#
#     def __init__(self,out_chan):
#         super(StemBlock, self).__init__()
#         self.conv = ConvBNReLU(3, out_chan // 2, 3, stride=2, use_relu=False)
#         self.left = nn.Sequential(
#             ConvBNReLU(32, 16, 1, stride=1, padding=0),
#             ConvBNReLU(16, 32, 3, stride=2),
#         )
#         self.right = nn.MaxPool2d(
#             kernel_size=3, stride=2, padding=1, ceil_mode=False)
#         self.fuse = ConvBNReLU(64, out_chan, 3, stride=1)
#
#     def forward(self, x):
#         feat = self.conv(x)
#         feat_left = self.left(feat)
#         feat_right = self.right(feat)
#         feat = torch.cat([feat_left, feat_right], dim=1)
#         feat = self.fuse(feat)
#         return feat

class StemBlock(nn.Module):

    def __init__(self,out_chan):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, out_chan // 2, 3, stride=2, use_relu=False)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, out_chan, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class ConvBNReLU1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU1, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class FeatureRefinementHead(nn.Module):
    def __init__(self, decode_channels=64, num_classes=19):
        super().__init__()
        # self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        # self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.eps = 1e-8
        self.post_conv = ConvBNReLU1(decode_channels, decode_channels, kernel_size=3)

        self.mp = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.ap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.class_conv = Conv(decode_channels, num_classes, kernel_size=1)
        self.act = nn.ReLU6()

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # weights = nn.ReLU()(self.weights)
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        # x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        # x = self.pre_conv(res) + x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        mp = self.mp(x) * x
        ap = self.ap(x) * x
        x = mp + ap
        print(x.shape)
        x = self.proj(x) + shortcut
        x = self.act(x)
        x = self.class_conv(x)
        print(x.shape)
        return x
