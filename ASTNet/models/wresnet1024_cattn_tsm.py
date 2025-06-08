import logging
import torch
import torch.nn as nn
from models.wider_resnet import wresnet
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights
from models.mem_module import MemModule

logger = logging.getLogger(__name__)


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=False):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info('=> ' + self.model_name + '_1024: (CATTN + TSM) - Ped2')

        self.wrn38 = wresnet(config, self.model_name, pretrained=pretrained)  # wrn38(config, pretrained=True)

        channels = [4096, 2048, 1024, 512, 256, 128]

        self.conv_x8 = nn.Conv2d(channels[0] * frames, channels[1], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[4] * frames, channels[4], kernel_size=1, bias=False)
        self.conv_x1 = nn.Conv2d(channels[5] * frames, channels[5], kernel_size=1, bias=False)

        self.up8 = ConvTransposeBnRelu(channels[1], channels[2], kernel_size=2)   # 2048          -> 1024
        #self.up4 = ConvTransposeBnRelu(channels[2] + channels[4], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        #self.up2 = ConvTransposeBnRelu(channels[3] + channels[5], channels[4], kernel_size=2)   # 512   +   128 -> 256
        self.up4 = ConvTransposeBnRelu(channels[2], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        self.up2 = ConvTransposeBnRelu(channels[3], channels[4], kernel_size=2)   # 512   +   128 -> 256

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left')

        self.attn8 = ChannelAttention(channels[2])
        self.attn4 = ChannelAttention(channels[3])
        self.attn2 = ChannelAttention(channels[4])

        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=1, padding=0),
            ConvBnRelu(channels[5], channels[5], kernel_size=3, padding=1),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)
        self.mem_rep8 = MemModule(mem_dim=2000, fea_dim=2048, shrink_thres =0.0025)
        self.mem_rep2 = MemModule(mem_dim=400, fea_dim=256, shrink_thres =0.0025)
        self.mem_rep1 = MemModule(mem_dim=800, fea_dim=128, shrink_thres =0.0025)
    def forward(self, x):
        x1s, x2s, x8s = [], [], []
        for xi in x:
            x1, x2, x8 = self.wrn38(xi)
            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)

        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))

        left = self.tsm_left(x8)
        x8 = x8 + left
        
        res_mem8 = self.mem_rep8(x8)
        att = []
        f8, att_8 = res_mem8['output'], res_mem8['att']
        att.append(att_8)

        #res_mem2 = self.mem_rep2(x2)
        #f2 = res_mem2['output']

        #res_mem1 = self.mem_rep1(x1)
        #f1 = res_mem1['output']

        x = self.up8(f8)
        x = self.up8(x8)
        x = self.attn8(x)

        #x = self.up4(torch.cat([f2, x], dim=1))
        x = self.up4(x)
        x = self.attn4(x)

        #x = self.up2(torch.cat([f1, x], dim=1))
        x = self.up2(x)
        x = self.attn2(x)

        return self.final(x), att

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//reduction, input_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8, direction='left'):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.direction = direction

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, direction=self.direction)
        return x

    @staticmethod
    def shift(x, n_segment=4, fold_div=8, direction='left'):
        bz, nt, h, w = x.size()
        c = nt // n_segment
        x = x.view(bz, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        if direction == 'left':
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        elif direction == 'right':
            out[:, 1:, :fold] = x[:, :-1, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        else:
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(bz, nt, h, w)
