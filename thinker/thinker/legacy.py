import torch
from torch import nn
from torch.nn import functional as F
from thinker import util
from thinker.core.module import ResBlock

class AFrameEncoderLegacy(nn.Module):
    def __init__(self,
                 input_shape, 
                 flags,
                 downpool=False, 
                 firstpool=False,    
                 out_size=256,
                 see_double=False,
                 ):
        
        super(AFrameEncoderLegacy, self).__init__()
        self.input_shape = input_shape
        self.downpool = downpool
        self.out_size = out_size

        if downpool:
            # see the frame directly; we need to have frame encoding
            self.frame_encoder = FrameEncoder(
                input_shape=input_shape,
                num_actions=None,
                downscale_c=2,
                size_nn=1,
                concat_action=False,
                grayscale=False,
            )
            input_shape = self.frame_encoder.out_shape

        # following code is from Torchbeast, which is the same as Impala deep model
        in_channels = input_shape[0]
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        for num_ch in [64, 64, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            # feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        # out shape after conv is: (num_ch, input_shape[1], input_shape[2])
        core_out_size = num_ch * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(nn.Linear(core_out_size, self.out_size), nn.ReLU())

    def forward(self, x):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""
        assert x.dtype in [torch.float, torch.float16]
        if self.downpool:
            x = self.frame_encoder(x, actions=None)
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = torch.flatten(x, start_dim=1)
        x = self.fc(F.relu(x))
        return x
    
class FrameEncoder(nn.Module):
    def __init__(
        self,
        num_actions,
        input_shape,
        size_nn=1,
        downscale_c=2,
        concat_action=True,
        decoder=False,
        frame_copy=False,
        grayscale=False,
        disable_bn=False,
    ):
        super(FrameEncoder, self).__init__()
        self.num_actions = num_actions
        self.size_nn = size_nn
        self.downscale_c = downscale_c
        self.decoder = decoder
        self.frame_copy = frame_copy
        self.grayscale = grayscale
        frame_channels, h, w = input_shape
        self.concat_action = concat_action

        if self.concat_action:
            in_channels = frame_channels + num_actions
        else:
            in_channels = frame_channels

        n_block = 1 * self.size_nn
        out_channels = int(128 // downscale_c)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        res = [
            ResBlock(inplanes=out_channels, disable_bn=disable_bn)
            for _ in range(n_block)
        ]  # Deep: 2 blocks here
        self.res1 = nn.Sequential(*res)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        res = [
            ResBlock(inplanes=out_channels * 2, disable_bn=disable_bn)
            for _ in range(n_block)
        ]  # Deep: 3 blocks here
        self.res2 = nn.Sequential(*res)
        self.avg1 = nn.AvgPool2d(3, stride=2, padding=1)
        res = [
            ResBlock(inplanes=out_channels * 2, disable_bn=disable_bn)
            for _ in range(n_block)
        ]  # Deep: 3 blocks here
        self.res3 = nn.Sequential(*res)
        self.avg2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.out_shape = (
            out_channels * 2,
            h // 16 + int((h % 16) > 0),
            w // 16 + int((w % 16) > 0),
        )

        if decoder:
            d_conv = [
                ResBlock(inplanes=out_channels * 2, disable_bn=disable_bn)
                for _ in range(n_block)
            ]
            kernel_sizes = [4, 4, 4, 4]
            conv_channels = [
                frame_channels
                if not self.frame_copy
                else (3 if not self.grayscale else 1),
                out_channels,
                out_channels * 2,
                out_channels * 2,
                out_channels * 2,
            ]
            for i in range(4):
                if i in [1, 3]:
                    d_conv.extend(
                        [
                            ResBlock(
                                inplanes=conv_channels[4 - i], disable_bn=disable_bn
                            )
                            for _ in range(n_block)
                        ]
                    )
                d_conv.append(nn.ReLU())
                d_conv.append(
                    nn.ConvTranspose2d(
                        conv_channels[4 - i],
                        conv_channels[4 - i - 1],
                        kernel_size=kernel_sizes[i],
                        stride=2,
                        padding=1,
                    )
                )
            self.d_conv = nn.Sequential(*d_conv)

    def forward(self, x, actions=None, flatten=False):
        """
        Args:
          x (tensor): frame with shape B, C, H, W
          action (tensor): action with shape B, num_actions (in one-hot)
        """
        assert x.dtype in [torch.float, torch.float16]
        if flatten:
            input_shape = x.shape
            x = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])            
        if self.concat_action:
            if flatten:
                actions = actions.view(
                    (actions.shape[0] * actions.shape[1],) + actions.shape[2:]
                )
            actions = (
                actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])
            )
            x = torch.concat([x, actions], dim=1)
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.avg1(x)
        x = self.res3(x)
        z = self.avg2(x)
        if flatten:
            z = z.view(input_shape[:2] + z.shape[1:])
        return z

    def decode(self, z, flatten=False):
        """
        Args:
          z (tensor): encoding with shape B, *
        """
        if flatten:
            input_shape = z.shape
            z = z.view((z.shape[0] * z.shape[1],) + z.shape[2:])
        # x = checkpoint_sequential(self.d_conv, segments=1, input=z)
        x = self.d_conv(z)
        if flatten:
            x = x.view(input_shape[:2] + x.shape[1:])
        return x


class ShallowAFrameEncoder(nn.Module):
    # shallow processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 out_size=256,
                 downscale=True):
        super(ShallowAFrameEncoder, self).__init__()
        self.input_shape = input_shape
        self.out_size = out_size

        c, h, w = self.input_shape
        compute_hw = lambda h, w, k, s: ((h - k) // s + 1,  (h - k) // s + 1)
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4 if downscale else 1)
        h, w = compute_hw(h, w, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2 if downscale else 1)
        h, w = compute_hw(h, w, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1 if downscale else 1)
        h, w = compute_hw(h, w, 3, 1)
        fc_in_size = h * w * 64
        # Fully connected layer.
        self.fc = nn.Linear(fc_in_size, out_size)

    def forward(self, x):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""

        assert x.dtype in [torch.float, torch.float16]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x