import  torch
import  torch.nn as nn
import torch.nn.functional as F

from attention import AttentionConv, NonLocalAttention




# OPS is a set of layers with same input/output channel.

OPS = {
    # 'none':         lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'attention_3*3_group_4': lambda C, stride, affine: Attention(C, C, 3, stride, 1, 4, affine=affine),
    'attention_5*5_group_4': lambda C, stride, affine: Attention(C, C, 5, stride, 2, 4, affine=affine),
    'attention_7*7_group_4': lambda C, stride, affine: Attention(C, C, 7, stride, 3, 4, affine=affine),
    'attention_3*3_group_8': lambda C, stride, affine: Attention(C, C, 3, stride, 1, 8, affine=affine),
    'attention_5*5_group_8': lambda C, stride, affine: Attention(C, C, 5, stride, 2, 8, affine=affine),
    'attention_7*7_group_8': lambda C, stride, affine: Attention(C, C, 7, stride, 3, 8, affine=affine),
    'NonLocal_Attention':lambda C, stride, affine: NonLocal_Attention(C,None, stride,affine),
}

class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups,affine=False, bias=False):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param affine:
        """
        super(Attention, self).__init__()
        self.stride = stride

        self.op1 = nn.Sequential(
            nn.Conv2d(C_in, C_in//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_in//2),
            nn.ReLU(),
        )
        self.op = nn.Sequential(
            AttentionConv(C_in//2, C_out//2, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(C_out//2, affine),
            nn.ReLU(),
        )
        self.op2 = nn.Sequential(
            nn.Conv2d(C_out//2, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(C_out)
            )


    def forward(self, x):
        out = self.op1(x)
        out = self.op(out)
        res = self.op2(out)
        if self.stride == 2:
            res = F.avg_pool2d(res, (self.stride, self.stride))
        out = res + self.shortcut(x)
        return out


class NonLocal_Attention(nn.Module):
    def __init__(self, in_channels, inter_channel , stride , affine=False):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param affine:
        """
        super(NonLocal_Attention, self).__init__()
        self.c = in_channels
        self.inter_c = inter_channel
        self.stride = stride
        self.op = NonLocalAttention(in_channels // 2, in_channels // 2)
        self.op1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )

        self.op2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels)
            )


    def forward(self, x):
        # print(x.size())
        out = self.op1(x)
        out = self.op(out)
        res = self.op2(out)
        if self.stride == 2:
            res = F.avg_pool2d(res, (self.stride, self.stride))
        # print(x.size())
        out1 = res + self.shortcut(x)
        return out1


class ReLUConvBN(nn.Module):
    """
    Stack of relu-conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param affine:
        """
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """
    relu-dilated conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 2/4
        :param dilation: 2
        :param affine:
        """
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """
    implemented separate convolution via pytorch groups parameters
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 1/2
        :param affine:
        """
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """
    zero by stride
    """
    def __init__(self, stride):
        super(Zero, self).__init__()

        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """
    reduce feature maps height/width by half while keeping channel same
    """

    def __init__(self, C_in, C_out, affine=True):
        """

        :param C_in:
        :param C_out:
        :param affine:
        """
        super(FactorizedReduce, self).__init__()

        assert C_out % 2 == 0

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)

        # x: torch.Size([32, 32, 32, 32])
        # conv1: [b, c_out//2, d//2, d//2]
        # conv2: []
        # out: torch.Size([32, 32, 16, 16])

        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
