import torch
from torch import nn
from torch.nn import functional as F

def simple_mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    zero_init=False,
    norm=True,
):
    """MLP layers
    args:
        input_size (int): dim of inputs
        layer_sizes (list): dim of hidden layers
        output_size (int): dim of outputs
        init_zero (bool): zero initialization for the last layer (including w and b).
            This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1], momentum=momentum))
            layers.append(act())
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    if zero_init:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        zero_init=False,
        norm=False,
        skip_connection=False,
    ):
        super().__init__()

        self.skip_connection = skip_connection

        sizes = [input_size] + layer_sizes + [output_size]
        self.layer_n = len(layer_sizes) + 1
        self.layers = nn.ModuleList()
        self.act = activation()
        self.output_act = output_activation()
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            if self.skip_connection and i >= 1:
                in_size += input_size
            layer = [nn.Linear(in_size, out_size)]
            if norm:
                layer.append(nn.BatchNorm1d(out_size, momentum=momentum))
            self.layers.append(nn.Sequential(*layer))

        if zero_init:
            self.layers[-1][0].weight.data.fill_(0)
            self.layers[-1][0].bias.data.fill_(0)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < self.layer_n - 1:
                out = self.act(out)
            else:
                out = self.output_act(out)
            if self.skip_connection and i < self.layer_n - 1:
                out = torch.cat((out, x), dim=-1)
        return out
        
def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes, outplanes=None, stride=1, downsample=None, disable_bn=False
    ):
        super().__init__()
        if outplanes is None:
            outplanes = inplanes
        if disable_bn:
            norm_layer = nn.Identity
        else:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        self.skip_conv = outplanes != inplanes
        self.stride = stride
        if outplanes != inplanes:
            if downsample is None:
                self.conv3 = conv1x1(inplanes, outplanes)
            else:
                self.conv3 = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip_conv:
            out += self.conv3(identity)
        else:
            out += identity
        out = self.relu(out)
        return out
    
class OneDResBlock(nn.Module):
    def __init__(self, hidden_size, norm=False):
        super(OneDResBlock, self).__init__()
        self.norm = norm
        if norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)        
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out = F.relu(self.norm1(self.linear1(x))) if self.norm else F.relu(self.linear1(x))
        out = F.relu(self.norm2(self.linear2(out))) if self.norm else F.relu(self.linear2(out))
        out = out + x  # Skip connection
        return out

def guassian_kl_div(tar_mean, tar_log_var, mean, log_var, reduce="sum"):
    tar_var, var = torch.exp(tar_log_var), torch.exp(log_var)
    tar_log_std, log_std = tar_log_var / 2, log_var / 2
    kl = log_std - tar_log_std + (tar_var + (tar_mean - mean).pow(2)) / (2 * var) - 0.5
    if reduce == "sum":
        return torch.sum(kl, dim=-1)
    else:
        return torch.mean(kl, dim=-1)
    
def tile_and_concat_tensors(tensor_list):
    # Step 1: Determine the maximum H and W
    max_H = max(tensor.shape[2] for tensor in tensor_list)
    max_W = max(tensor.shape[3] for tensor in tensor_list)

    tiled_tensors = []

    for tensor in tensor_list:
        # Step 2: Calculate the tiling factor for each dimension
        tile_factor_H = max_H // tensor.shape[2]
        tile_factor_W = max_W // tensor.shape[3]

        # Step 3: Tile the tensor
        tiled_tensor = tensor.repeat(1, 1, tile_factor_H, tile_factor_W)

        # Ensure that the tensor is cropped or padded to match the exact size (max_H, max_W)
        if tiled_tensor.shape[2] < max_H or tiled_tensor.shape[3] < max_W:
            padding_H = max_H - tiled_tensor.shape[2]
            padding_W = max_W - tiled_tensor.shape[3]
            tiled_tensor = torch.nn.functional.pad(tiled_tensor, (0, padding_W, 0, padding_H), mode='constant', value=0)
        
        # Then, crop if the tensor is too large
        tiled_tensor = tiled_tensor[:, :, :max_H, :max_W]

        tiled_tensors.append(tiled_tensor)

    # Step 4: Concatenate all tensors along the channel dimension
    concatenated_tensor = torch.cat(tiled_tensors, dim=1)

    return concatenated_tensor    