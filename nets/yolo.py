from collections import OrderedDict

import torch
from torch import nn

from nets.darknet import darknet53


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# class SeparableConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
#         super(SeparableConvBlock, self).__init__()
#         if out_channels is None:
#             out_channels = in_channels
#
#         self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
#         self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#
#         self.norm = norm
#         if self.norm:
#             self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
#
#         self.activation = activation
#         if self.activation:
#             self.swish = Swish()
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)
#
#         if self.norm:
#             x = self.bn(x)
#
#         if self.activation:
#             x = self.swish(x)
#
#         return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(in_filters, out_filter):
    m = nn.Sequential(
        # SeparableConvBlock(in_filters, filters_list[0], 1),
        # SeparableConvBlock(filters_list[0], filters_list[1], 3),
        # SeparableConvBlock(filters_list[1], filters_list[0], 1),
        # SeparableConvBlock(filters_list[0], filters_list[1], 3),
        # SeparableConvBlock(filters_list[1], filters_list[0], 1),
        # conv2d(filters_list[0], filters_list[1], 3),
        # nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
        conv2d(in_filters, in_filters, 3),
        nn.Conv2d(in_filters, out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m



class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False, epsilon=1e-4, attention=True):
        super(YoloBody, self).__init__()
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        self.num_channels = len(anchors_mask[0]) * (num_classes + 5)

        self.epsilon = epsilon
        self.swish = Swish()

        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.p3_upsample = Upsample(512, 256)
        self.p4_upsample = Upsample(1024, 512)

        self.p4_downsample = conv2d(256, 512, 3, 2)
        self.p5_downsample = conv2d(512, 1024, 3, 2)

        self.conv4_up = conv2d(512, 512, 3)
        self.conv3_up = conv2d(256, 256, 3)

        self.conv4_down = conv2d(512, 512, 3)
        self.conv5_down = conv2d(1024, 1024, 3)

        # 计算yolo_head的输出通道数
        self.p3_out_0 = make_last_layers(256, self.num_channels)
        self.p4_out_0 = make_last_layers(512, self.num_channels)
        self.p5_out_0 = make_last_layers(1024, self.num_channels)

    def forward(self, x):
        p3_in, p4_in, p5_in = self.backbone(x)

        # 简单的注意力机制，用于确定更关注p5_in还是p4_in
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))

        # 简单的注意力机制，用于确定更关注p3_in还是p4_td
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

        # 简单的注意力机制，用于确定更关注p4_in还是p4_td还是p3_out
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

        # 简单的注意力机制，用于确定更关注p5_in还是p4_out
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * self.p5_downsample(p4_out)))

        p3_out = self.p3_out_0(p3_out)
        p4_out = self.p4_out_0(p4_out)
        p5_out = self.p5_out_0(p5_out)

        return p5_out, p4_out, p3_out







# from collections import OrderedDict
#
# import torch
# import torch.nn as nn
#
# from nets.darknet import darknet53
#
# def conv2d(filter_in, filter_out, kernel_size):
#     pad = (kernel_size - 1) // 2 if kernel_size else 0
#     return nn.Sequential(OrderedDict([
#         ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
#         ("bn", nn.BatchNorm2d(filter_out)),
#         ("relu", nn.LeakyReLU(0.1)),
#     ]))
#
# #------------------------------------------------------------------------#
# #   make_last_layers里面一共有七个卷积，前五个用于提取特征。
# #   后两个用于获得yolo网络的预测结果
# #------------------------------------------------------------------------#
# def make_last_layers(filters_list, in_filters, out_filter):
#     m = nn.Sequential(
#         conv2d(in_filters, filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
#     )
#     return m
#
# class YoloBody(nn.Module):
#     def __init__(self, anchors_mask, num_classes, pretrained = False):
#         super(YoloBody, self).__init__()
#         #---------------------------------------------------#
#         #   生成darknet53的主干模型
#         #   获得三个有效特征层，他们的shape分别是：
#         #   52,52,256
#         #   26,26,512
#         #   13,13,1024
#         #---------------------------------------------------#
#         self.backbone = darknet53()
#         if pretrained:
#             self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))
#
#         #---------------------------------------------------#
#         #   out_filters : [64, 128, 256, 512, 1024]
#         #---------------------------------------------------#
#         out_filters = self.backbone.layers_out_filters
#
#         #------------------------------------------------------------------------#
#         #   计算yolo_head的输出通道数，对于voc数据集而言
#         #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
#         #------------------------------------------------------------------------#
#         self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
#
#         self.last_layer1_conv       = conv2d(512, 256, 1)
#         self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
#         self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
#
#         self.last_layer2_conv       = conv2d(256, 128, 1)
#         self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
#         self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
#
#     def forward(self, x):
#         #---------------------------------------------------#
#         #   获得三个有效特征层，他们的shape分别是：
#         #   52,52,256；26,26,512；13,13,1024
#         #---------------------------------------------------#
#         x2, x1, x0 = self.backbone(x)
#
#         #---------------------------------------------------#
#         #   第一个特征层
#         #   out0 = (batch_size,255,13,13)
#         #---------------------------------------------------#
#         # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
#         out0_branch = self.last_layer0[:5](x0)
#         out0        = self.last_layer0[5:](out0_branch)
#
#         # 13,13,512 -> 13,13,256 -> 26,26,256
#         x1_in = self.last_layer1_conv(out0_branch)
#         x1_in = self.last_layer1_upsample(x1_in)
#
#         # 26,26,256 + 26,26,512 -> 26,26,768
#         x1_in = torch.cat([x1_in, x1], 1)
#         #---------------------------------------------------#
#         #   第二个特征层
#         #   out1 = (batch_size,255,26,26)
#         #---------------------------------------------------#
#         # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
#         out1_branch = self.last_layer1[:5](x1_in)
#         out1        = self.last_layer1[5:](out1_branch)
#
#         # 26,26,256 -> 26,26,128 -> 52,52,128
#         x2_in = self.last_layer2_conv(out1_branch)
#         x2_in = self.last_layer2_upsample(x2_in)
#
#         # 52,52,128 + 52,52,256 -> 52,52,384
#         x2_in = torch.cat([x2_in, x2], 1)
#         #---------------------------------------------------#
#         #   第一个特征层
#         #   out3 = (batch_size,255,52,52)
#         #---------------------------------------------------#
#         # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
#         out2 = self.last_layer2(x2_in)
#         return out0, out1, out2