import torch
from torch import nn
import torch.nn.functional as F


# 它定义了一个包含卷积、标准化和 LeakyReLU 激活函数的神经网络层
class ConvNormLReLU(nn.Sequential):
    # ConvNormLReLU 类的构造函数
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            # 在构造函数中，首先根据给定的参数 pad_mode（表示 padding 的方式）选择适当的 padding 层，并将其作为第一个组件添加到神经网络层中。然后，通过调用 nn.Conv2d 构造函数，在
            # pad_layer[pad_mode](padding) 的基础上继续添加一个卷积层，其中的参数包括输入通道数 in_ch、输出通道数 out_ch、卷积核大小 kernel_size、步长
            # stride、padding 大小 padding、groups 数量 groups 和是否带偏置项 bias。 接着，ConvNormLReLU 层还包含一个 nn.GroupNorm
            # 组件，用于进行标准化操作。其中，num_groups=1 表示不分组，即整个 feature map 作为一个 batch 进行归一化。num_channels=out_ch 表示要对输出通道数为
            # out_ch 的 feature map 进行标准化操作。 最后，还需要添加一个 LeakyReLU 激活函数，用于增强非线性映射能力。
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []

        # 如果扩张因子 expansion_ratio 不等于 1，则加入一个 1×1 的卷积层，进行通道数的调整。
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        # 深度可分离卷积层（depthwise convolutional layer），其中 bottleneck 参数指定了中间层的通道数，
        # groups 参数设置为 bottleneck，表示按照通道分组卷积运算。
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))

        # pw
        # 1×1 卷积层（pointwise convolutional layer），其中 out_ch 参数指定了输出通道数。
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))

        # 组归一化层（group normalization layer）
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        # BlockA: 对输入图像进行第一次卷积、归一化和激活函数处理；
        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        # BlockB: 对经过BlockA处理后得到的特征图进行第二次卷积、归一化和激活函数处理；
        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        # BlockC: 一系列使用倒置残差块的卷积操作
        # 其中每个倒置残差块包含了1个11卷积、1个独立卷积和1个11卷积，可以进一步提高图像质量；
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        # BlockD: 对BlockC的输出进行第三次卷积、归一化和激活函数处理；
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        # BlockE: 对BlockD的输出进行最后一轮的卷积、归一化和激活函数处理，得到最后的特征图；
        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        # OutputLayer：输出层，将最后的特征图转换为一张生成的图像。
        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    # forward()
    # 方法定义了模型的前向计算过程，即将输入数据流经网络，输出预测结果。
    def forward(self, input, align_corners=True):

        # 首先，将输入图像传入 block_a 层进行卷积、归一化和激活函数处理，得到处理后的 out；
        # 然后，将 out 传递给 block_b 层进行相同的处理操作；
        # 接着，将 out 传递给 block_c 层进行一系列使用倒置残差块的卷积操作的块，进一步提高图像质量；
        # 随后，调用 F.interpolate 对处理后的 out 进行上采样操作，将其增大为原始输入大小的一半（其中 align_corners 参数为 True 表示对齐边界像素）；
        # 将上一步的输出传递给 block_d 层进行相同的处理操作；
        # 再次调用 F.interpolate 对处理后的 out 进行上采样操作，将其还原为原来的大小（同样，align_corners=True 表示对齐边界像素）；
        # 最后，将上一步的输出传递给 block_e 层进行最后一轮卷积、归一化和激活函数处理操作；
        # 最终，将处理后的图像通过 out_layer 输出。

        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out
