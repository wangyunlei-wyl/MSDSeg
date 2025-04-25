import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from blocks import *
from benchmark import *


def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


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


class MSDSeg(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)

    def __init__(self, num_classes=19, dilations=[1, 4, 8],
                 pretrained='', change_num_classes=False):
        super(MSDSeg, self).__init__()
        self.stem = StemBlock(out_chan=32)
        self.stage8 = nn.Sequential(
            mod_Inception_block(32, 64, dilation=dilations, branch_planes=16, stride=2),
            mod_Inception_block(64, 64, dilation=dilations, branch_planes=32),
            mod_Inception_block(64, 64, dilation=dilations, branch_planes=32),
            mod_Inception_block(64, 64, dilation=dilations, branch_planes=32),
            mod_Inception_block(64, 64, dilation=dilations, branch_planes=32),
            mod_Inception_block(64, 64, dilation=dilations, branch_planes=32),
        )
        self.stage16 = nn.Sequential(
            mod_Inception_block(64, 128, dilation=dilations, branch_planes=32, stride=2),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
            mod_Inception_block(128, 128, dilation=dilations, branch_planes=64),
        )
        self.fu16 = Bag(128, 64, 64)
        self.fu8 = Bag(64, 32, 64)
        self.frh = FeatureRefinementHead(64, num_classes)
        # self.apply(self.weight_init)
        if pretrained != "":
            print('use pretrain model {}'.format(pretrained))
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)

    def forward(self, m):
        input_shape = m.shape[-2:]
        x4 = self.stem(m)  # [2, 64, 128, 256]  2
        x8 = self.stage8(x4)  # torch.Size([1, 64, 64, 128])  8
        x16 = self.stage16(x8)  # torch.Size([1, 128, 32, 64])  16
        fu16 = self.fu16(F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False), x8)
        fu8 = self.fu8(F.interpolate(fu16, size=x4.shape[-2:], mode='bilinear', align_corners=False), x4)
        out = F.interpolate(self.frh(fu8), size=m.shape[-2:], mode='bilinear', align_corners=False)
        return out



algc = False


class sir(nn.Module):
    def __init__(
            self, in_channels, stride=1, BatchNorm=nn.BatchNorm2d, gw=16):
        super(sir, self).__init__()
        bn_mom = 0.1
        self.stride = stride
        branch_planes = in_channels * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_planes, kernel_size=3, stride=stride, padding=1, groups=gw, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.cs = CS(gw)
        self.branch1 = nn.Sequential(
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 1), stride=1, groups=gw,
                      padding=(1, 0), dilation=1, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(1, 3), stride=1, groups=gw,
                      padding=(0, 1), dilation=1, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
        )
        self.attention = ECA(branch_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(branch_planes, in_channels, kernel_size=1, groups=16, bias=False),
            BatchNorm(in_channels, momentum=bn_mom),
        )
        self.shortcut = Shortcut(in_channels, in_channels, stride=stride)

    def forward(self, x):
        if self.stride == 2:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        x = self.conv1(x)
        x = self.branch1(x)
        x = self.attention(x)
        x = self.conv2(x)
        out = self.relu(x + shortcut)
        out = self.cs(out)
        return out


class mod_Inception_block(nn.Module):
    def __init__(
            self, in_channels, outplanes, branch_planes=24, gw=16, stride=1, dilation=[1, 3, 5],
            BatchNorm=nn.BatchNorm2d):
        super(mod_Inception_block, self).__init__()
        bn_mom = 0.1
        self.stride = stride
        self.scale_process = nn.Sequential(
            nn.Conv2d(in_channels, branch_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 1), stride=1, groups=gw,
                      padding=(dilation[0], 0), dilation=dilation[0], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(1, 3), stride=1, groups=gw,
                      padding=(0, dilation[0]), dilation=dilation[0], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 1), stride=1, groups=gw,
                      padding=(dilation[1], 0), dilation=dilation[1], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(1, 3), stride=1, groups=gw,
                      padding=(0, dilation[1]), dilation=dilation[1], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(3, 1), stride=1, groups=gw,
                      padding=(dilation[2], 0), dilation=dilation[2], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=(1, 3), stride=1, groups=gw,
                      padding=(0, dilation[2]), dilation=dilation[2], bias=False),
            BatchNorm(branch_planes, momentum=bn_mom),
        )

        # self.attention = CBAM(branch_planes * 3)
        self.attention = SEModule(branch_planes * 3, outplanes)
        self.compression = nn.Sequential(
            nn.Conv2d(branch_planes * 3, outplanes, kernel_size=1, bias=False),
            BatchNorm(outplanes, momentum=bn_mom),
        )
        self.relu = nn.ReLU(inplace=True)
        self.cs = CS(gw)
        self.shortcut = Shortcut(in_channels, out_channels=outplanes, stride=stride)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        if self.stride == 2:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        x = self.scale_process(x)
        scale_list = []
        scale_list.append(self.branch1(x))
        scale_list.append(self.branch2(x) + scale_list[0])
        scale_list.append(self.branch3(x) + scale_list[1])
        se = self.attention(torch.cat(scale_list, 1))
        out = self.relu(self.compression(se) + shortcut)
        out = self.cs(out)
        return out


def calculate_flops():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, ActivationCountAnalysis
    model1 = MSDSeg().eval()
    # print(model1)
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23, get_ddrnet_23slim
    x = torch.randn(1, 3, 512, 1024)
    model2 = get_ddrnet_23().eval()
    for model in [model1, model2]:
        flops = FlopCountAnalysis(model, x)
        print(flop_count_table(flops))


def calculate_params(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = model.parameters()
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    return params, params2


def cityscapes_speed_test():
    print("cityscapes speed test")
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    model = MSDSeg()
    ddrnet23 = get_ddrnet_23()
    x = torch.randn(1, 3, 1024, 2048)
    ts = []
    ts.extend(benchmark_eval([model, ddrnet23], x, True))
    print(ts)


def camvid_speed_test():
    print("camvid speed test")
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    model = MSDSeg()
    ddrnet23 = get_ddrnet_23()
    x = torch.randn(1, 3, 720, 960)
    ts = []
    ts.extend(benchmark_eval([model, ddrnet23], x, True))
    print(ts)


if __name__ == "__main__":
    # cityscapes_speed_test()
    calculate_flops()
    # dilation_speed_test()
    # block_speed_test()
    # calculate_params(model)
