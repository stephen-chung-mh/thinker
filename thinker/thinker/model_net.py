from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker import util

OutNetOut = namedtuple(
    "OutNetOut",
    [
        "rs", "r_enc_logits", "dones", "done_logits", "vs", "v_enc_logits", "logits"
    ],
)
SRNetOut = namedtuple(
    "SRNetOut",
    ["rs", "r_enc_logits", "dones", "done_logits", "xs", "hs", "state"],
)
VPNetOut = namedtuple(
    "VPNetOut",
    ["rs", "r_enc_logits", "dones", "done_logits", "vs", "v_enc_logits",
        "logits", "hs", "pred_zs", "true_zs", "state",
    ],
)
DualNetOut = namedtuple(
    "DualNetOut", ["rs", "dones", "vs", "logits", "xs", "hs", "zs", "state"]
)

def mlp(
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

class MLPWithSkipConnections(nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        zero_init=False,
        norm=True,
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

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.rv_tran = None

    def get_weights(self):
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        tensor = isinstance(next(iter(weights.values())), torch.Tensor)
        if not tensor:
            self.load_state_dict(
                {k: torch.tensor(v, device=device) for k, v in weights.items()}
            )
        else:
            self.load_state_dict({k: v.to(device) for k, v in weights.items()})

# Model Network

class FrameEncoder(nn.Module):
    def __init__(
        self,
        num_actions,
        input_shape,
        size_nn=1,
        downscale_c=2,
        concat_action=True,
        decoder=False,
        frame_stack_n=1,
        disable_bn=False,
    ):
        super(FrameEncoder, self).__init__()
        self.num_actions = num_actions
        self.size_nn = size_nn
        self.downscale_c = downscale_c
        self.decoder = decoder
        self.frame_stack_n = frame_stack_n
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
                if frame_stack_n == 1
                else frame_channels // frame_stack_n,
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
            actions = actions.view(
                (actions.shape[0] * actions.shape[1],) + actions.shape[2:]
            )
        if self.concat_action:
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
        x = self.d_conv(z)
        if flatten:
            x = x.view(input_shape[:2] + x.shape[1:])
        return x

class DynamicModel(nn.Module):
    def __init__(
        self,
        num_actions,
        inplanes,
        size_nn=1,
        outplanes=None,
        disable_half_grad=False,
        disable_bn=False,
    ):
        super(DynamicModel, self).__init__()
        self.num_actions = num_actions
        self.inplanes = inplanes
        self.size_nn = size_nn
        self.disable_half_grad = disable_half_grad
        if outplanes is None:
            outplanes = inplanes

        res = [
            ResBlock(
                inplanes=inplanes + num_actions,
                outplanes=outplanes,
                disable_bn=disable_bn,
            )
        ] + [
            ResBlock(inplanes=outplanes, disable_bn=disable_bn)
            for i in range(4 * self.size_nn)
        ]
        self.res = nn.Sequential(*res)
        self.outplanes = outplanes

    def forward(self, h, actions):
        x = h
        b, c, height, width = x.shape
        if self.training and not self.disable_half_grad:
            # no half-gradient for dreamer net
            x.register_hook(lambda grad: grad * 0.5)
        actions = (
            actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])
        )
        x = torch.concat([x, actions], dim=1)
        out = self.res(x)
        return out


class OutputNet(nn.Module):
    def __init__(
        self,
        num_actions,
        input_shape,
        value_clamp,
        enc_type,
        zero_init,
        size_nn,
        predict_v_pi=True,
        predict_r=True,
        predict_done=False,
        disable_bn=False,
    ):
        super(OutputNet, self).__init__()

        self.input_shape = input_shape
        self.size_nn = size_nn
        self.value_clamp = value_clamp
        self.enc_type = enc_type
        self.predict_v_pi = predict_v_pi
        self.predict_r = predict_r
        self.predict_done = predict_done

        assert self.enc_type in [0, 2, 3], "model encoding type can only be 0, 2, 3"

        c, h, w = input_shape
        if self.enc_type in [2, 3]:
            self.rv_tran = RVTran(enc_type=enc_type)
            out_n = self.rv_tran.encoded_n
        else:
            self.rv_tran = None
            out_n = 1

        layer_norm = nn.BatchNorm2d if not disable_bn else nn.Identity

        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=c // 2, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=c // 2, out_channels=c // 4, kernel_size=3, padding="same"
        )
        fc_in = h * w * (c // 4)

        if predict_v_pi:
            self.fc_logits = nn.Linear(fc_in, num_actions)
            self.fc_v = nn.Linear(fc_in, out_n)
            if zero_init:
                nn.init.constant_(self.fc_v.weight, 0.0)
                nn.init.constant_(self.fc_v.bias, 0.0)
                nn.init.constant_(self.fc_logits.weight, 0.0)
                nn.init.constant_(self.fc_logits.bias, 0.0)

        if predict_done:
            self.fc_done = nn.Linear(fc_in, 1)
            if zero_init:
                nn.init.constant_(self.fc_done.weight, 0.0)

        if predict_r:
            self.fc_r = nn.Linear(fc_in, out_n)
            if zero_init:
                nn.init.constant_(self.fc_r.weight, 0.0)
                nn.init.constant_(self.fc_r.bias, 0.0)

    def forward(self, h, predict_reward=True, state={}):
        x = h
        b = x.shape[0]
        state_ = {}
        x_ = F.relu(self.conv1(x))
        x_ = F.relu(self.conv2(x_))
        x_ = torch.flatten(x_, start_dim=1)
        x_v, x_logits, x_done = x_, x_, x_

        if self.predict_v_pi:
            logits = self.fc_logits(x_logits)
            if self.enc_type in [2, 3]:
                v_enc_logit = self.fc_v(x_v)
                v_enc_v = F.softmax(v_enc_logit, dim=-1)
                v = self.rv_tran.decode(v_enc_v)
            else:
                v_enc_logit = None
                v = self.fc_v(x_v).squeeze(-1)
                if self.value_clamp is not None and self.value_clamp > 0:
                    v = torch.clamp(v, -self.value_clamp, self.value_clamp)
        else:
            v, v_enc_logit, logits = None, None, None

        if self.predict_done:
            done_logit = self.fc_done(x_done).squeeze(-1)
            done = (nn.Sigmoid()(done_logit) > 0.5).bool()
        else:
            done_logit, done = None, None

        if self.predict_r and predict_reward:
            x_r = x_
            r_out = self.fc_r(x_r)
            if self.enc_type in [2, 3]:
                r_enc_logit = r_out
                r_enc_v = F.softmax(r_enc_logit, dim=-1)
                r = self.rv_tran.decode(r_enc_v)
            else:
                r_enc_logit = None
                r = r_out.squeeze(-1)
        else:
            r, r_enc_logit = None, None
        out = OutNetOut(
            rs=r,
            r_enc_logits=r_enc_logit,
            dones=done,
            done_logits=done_logit,
            vs=v,
            v_enc_logits=v_enc_logit,
            logits=logits,
        )
        return out

class SRNet(nn.Module):
    def __init__(self, obs_shape, num_actions, flags, frame_stack_n=1):
        super(SRNet, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.enc_type = flags.model_enc_type
        self.size_nn = flags.model_size_nn
        self.downscale_c = flags.model_downscale_c
        self.frame_stack_n = frame_stack_n
        assert self.obs_shape[0] % self.frame_stack_n == 0, \
            f"obs channel {self.obs_shape[0]} should be divisible by frame stacking number {self.frame_stack_n}"        
        self.copy_n = self.obs_shape[0] // self.frame_stack_n
        self.encoder = FrameEncoder(
            num_actions=num_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=True,
            frame_stack_n=self.frame_stack_n,
        )
        self.hidden_shape = self.encoder.out_shape
        inplanes = self.hidden_shape[0]
        self.RNN = DynamicModel(
            num_actions=num_actions,
            inplanes=inplanes,
            size_nn=self.size_nn,
            disable_half_grad=True,
            disable_bn=self.flags.model_disable_bn,
        )
        if self.flags.reward_clip > 0:
            value_clamp = self.flags.reward_clip / (1 - self.flags.discounting)
        else:
            value_clamp = None
        self.out = OutputNet(
            num_actions=num_actions,
            input_shape=self.hidden_shape,
            value_clamp=value_clamp,
            enc_type=self.enc_type,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=False,
            predict_r=True,
            predict_done=self.flags.model_done_loss_cost > 0.0,
            disable_bn=self.flags.model_disable_bn,
        )
        self.rv_tran = self.out.rv_tran

    def forward(self, x, actions, one_hot=False):
        """
        Args:
            x(tensor): frames (float) with shape (B, C, H, W), in the form of s_t
            actions(tensor): action (int64) with shape (k+1, B, *), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            SRNetOut tuple with predicted rewards (rs), images (xs), done (dones) in the shape of (k, B, ...);
                in the form of y_{t+1}, y_{t+2}, ..., y_{t+k} and states with element in the shape of (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)
        h = self.encoder(x, actions[0])
        hs = [h.unsqueeze(0)]
        for t in range(1, k + 1):
            h = self.RNN(h=h, actions=actions[t])
            hs.append(h.unsqueeze(0))
        hs = torch.concat(hs, dim=0)

        state = {"sr_h": h}
        if len(hs) > 1:
            xs = self.encoder.decode(hs[1:], flatten=True)

            if self.frame_stack_n > 1:
                stacked_x = x
                stacked_xs = []
                for i in range(k):
                    stacked_x = torch.concat([stacked_x[:, self.copy_n:], xs[i]], dim=1)
                    stacked_xs.append(stacked_x)
                xs = torch.stack(stacked_xs, dim=0)
                state["last_x"] = stacked_x[:, self.copy_n:]   
       
        else:
            xs = None            
            if self.frame_stack_n > 1:
                state["last_x"] = x[:, self.copy_n:]     

        outs = []
        for t in range(1, k + 1):
            out = self.out(hs[t], predict_reward=True)
            outs.append(out)
        return SRNetOut(
            rs=util.safe_concat(outs, "rs", 0),
            r_enc_logits=util.safe_concat(outs, "r_enc_logits", 0),
            dones=util.safe_concat(outs, "dones", 0),
            done_logits=util.safe_concat(outs, "done_logits", 0),
            xs=xs,
            hs=hs,
            state=state,
        )

    def forward_single(self, action, state, one_hot=False):
        """
        Single unroll of the network with one action
        Args:
            action(tensor): action (int64) with shape (B, *)
            one_hot (bool): whether to the action use one-hot encoding
        """
        if not one_hot:
            action = F.one_hot(action, self.num_actions)
        h = self.RNN(h=state["sr_h"], actions=action)
        x = self.encoder.decode(h, flatten=False)
        if self.frame_stack_n > 1:
            x = torch.concat([state["last_x"], x], dim=1)

        out = self.out(h, predict_reward=True, state=state)
        state = {"sr_h": h}
        if self.frame_stack_n > 1:
            state["last_x"] = x[:, self.copy_n:]        
        
        return SRNetOut(
            rs=util.safe_unsqueeze(out.rs, 0),
            r_enc_logits=util.safe_unsqueeze(out.r_enc_logits, 0),
            dones=util.safe_unsqueeze(out.dones, 0),
            done_logits=util.safe_unsqueeze(out.done_logits, 0),
            xs=util.safe_unsqueeze(x, 0),
            hs=util.safe_unsqueeze(h, 0),
            state=state,
        )

class VPNet(nn.Module):
    def __init__(self, obs_shape, num_actions, flags):
        super(VPNet, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.enc_type = flags.model_enc_type
        self.size_nn = (
            flags.model_size_nn
        )  # size_nn: int to adjust for size of model net
        self.downscale_c = (
            flags.model_downscale_c
        )  # downscale_c: int to downscale number of channels; default=2
        self.use_rnn = (
            flags.wrapper_type != 2
        )  # dont use rnn if we have perfect dynamic
        self.receive_z = (
            flags.dual_net
        )  # rnn receives z only when we are using dual net
        self.predict_rd = (
            not flags.dual_net and flags.wrapper_type != 2
        )  # network also predicts reward and done if not dual net under non-perfect dynamic   

        self.encoder = FrameEncoder(
            num_actions=num_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=False,
        )
        self.hidden_shape = self.encoder.out_shape
        inplanes = self.hidden_shape[0]
        if self.use_rnn:
            self.RNN = DynamicModel(
                num_actions=num_actions,
                inplanes=inplanes * 2 if self.receive_z else inplanes,
                outplanes=inplanes,
                size_nn=self.size_nn,
                disable_half_grad=False,
                disable_bn=self.flags.model_disable_bn,
            )
        if self.flags.reward_clip > 0:
            value_clamp = self.flags.reward_clip / (1 - self.flags.discounting)
        else:
            value_clamp = None
        self.out = OutputNet(
            num_actions=num_actions,
            input_shape=self.hidden_shape,
            value_clamp=value_clamp,
            enc_type=self.enc_type,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=True,
            predict_r=self.predict_rd,
            predict_done=self.predict_rd and self.flags.model_done_loss_cost > 0.0,
            disable_bn=self.flags.model_disable_bn,
        )

        if not self.receive_z:
            self.h_to_z_conv = nn.Sequential(
                ResBlock(inplanes=inplanes, disable_bn=False),
                conv3x3(inplanes, inplanes),
            )
            self.z_to_h_conv = nn.Sequential(
                ResBlock(inplanes=inplanes, disable_bn=False),
                conv3x3(inplanes, inplanes),
            )

        self.rv_tran = self.out.rv_tran

    def h_to_z(self, h, flatten=False):
        if flatten:
            h_ = torch.flatten(h, 0, 1)
        else:
            h_ = h
        z = self.h_to_z_conv(h_)
        if flatten:
            z = z.view(h.shape[:2] + z.shape[1:])
        return z

    def z_to_h(self, z, flatten=False):
        if flatten:
            z_ = torch.flatten(z, 0, 1)
        else:
            z_ = z
        h = self.z_to_h_conv(z_)
        if flatten:
            h = h.view(z.shape[:2] + h.shape[1:])
        return h

    def forward(self, xs, actions, one_hot=False):
        """
        Args:
            xs(tensor): frames (uint8) with shape (k+1, B, C, H, W) in the form of s_t, s_{t+1}, ..., s_{t+k}
              or (1, B, C, H, W) / (B, C, H, W) in the form of s_t
            actions(tensor): action (int64) with shape (k+1, B, *), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            VPNetOut tuple with predicted values (vs), policies (logits) in the shape of (k+1, B, ...);
                in the form of y_{t}, y_{t+1}, y_{t+2}, ..., y_{t+k} and states with element in the shape of (B, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        if len(xs.shape) == 4:
            assert not self.receive_z
            xs = xs.unsqueeze(0)
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)
        zs = self.encoder(xs.detach(), actions[: xs.shape[0]], flatten=True)
        if self.use_rnn:
            if not self.receive_z:
                h = self.z_to_h(zs[0], flatten=False)
            else:
                h = torch.zeros(size=(b,) + self.hidden_shape, device=xs.device)
                rnn_in = torch.concat([h, zs[0]], dim=1)
                h = self.RNN(h=rnn_in, actions=actions[0])
            hs = [h.unsqueeze(0)]
            for t in range(1, k + 1):
                rnn_in = torch.concat([h, zs[t]], dim=1) if self.receive_z else h
                h = self.RNN(h=rnn_in, actions=actions[t])
                hs.append(h.unsqueeze(0))
            hs = torch.concat(hs, dim=0)
        else:
            hs = zs
            h = hs[-1]

        outs = []
        for t in range(0, k + 1):
            out = self.out(hs[t], predict_reward=t > 0)
            outs.append(out)

        if not self.receive_z:
            pred_zs = torch.concat([zs[[0]], self.h_to_z(hs[1:], flatten=True)], dim=0)
        else:
            pred_zs = zs

        state = {"vp_h": h}
        return VPNetOut(
            rs=util.safe_concat(outs[1:], "rs", 0),
            r_enc_logits=util.safe_concat(outs[1:], "r_enc_logits", 0),
            dones=util.safe_concat(outs[1:], "dones", 0),
            done_logits=util.safe_concat(outs[1:], "done_logits", 0),
            vs=util.safe_concat(outs, "vs", 0),
            v_enc_logits=util.safe_concat(outs, "v_enc_logits", 0),
            logits=util.safe_concat(outs, "logits", 0),
            hs=hs,
            true_zs=zs,
            pred_zs=pred_zs,
            state=state,
        )

    def forward_single(self, state, action, x=None, one_hot=False):
        """
        Single unroll of the network with one action
        Args:
            x(tensor): frame (float) with shape (B, *)
            action(tensor): action (int64) with shape (B, *)
            one_hot (bool): whether to the action use one-hot encoding
        """
        if not one_hot:
            action = F.one_hot(action, self.num_actions)
        if self.receive_z:
            z = self.encoder(x, action, flatten=False)
            rnn_in = torch.concat([state["vp_h"], z], dim=1)
        else:
            rnn_in = state["vp_h"]
        h = self.RNN(h=rnn_in, actions=action)
        out = self.out(h, predict_reward=True, state=state)
        state = {"vp_h": h}

        if not self.receive_z:
            pred_z = self.h_to_z(h, flatten=False)
        else:
            pred_z = z

        return VPNetOut(
            rs=util.safe_unsqueeze(out.rs, 0),
            r_enc_logits=util.safe_unsqueeze(out.r_enc_logits, 0),
            dones=util.safe_unsqueeze(out.dones, 0),
            done_logits=util.safe_unsqueeze(out.done_logits, 0),
            vs=util.safe_unsqueeze(out.vs, 0),
            v_enc_logits=util.safe_unsqueeze(out.v_enc_logits, 0),
            logits=util.safe_unsqueeze(out.logits, 0),
            hs=util.safe_unsqueeze(h, 0),
            true_zs=None,
            pred_zs=util.safe_unsqueeze(pred_z, 0),
            state=state,
        )

class ModelNet(BaseNet):
    def __init__(self, obs_space, num_actions, flags, frame_stack_n=1):
        super(ModelNet, self).__init__()
        self.rnn = False
        self.flags = flags
        self.obs_shape = obs_space.shape
        self.num_actions = num_actions
        self.enc_type = flags.model_enc_type
        self.size_nn = flags.model_size_nn
        self.dual_net = flags.dual_net
        
        self.register_buffer("norm_low", torch.tensor(obs_space.low))
        self.register_buffer("norm_high", torch.tensor(obs_space.high))

        self.vp_net = VPNet(self.obs_shape, num_actions, flags)
        self.hidden_shape = list(self.vp_net.hidden_shape)
        if self.dual_net:
            self.sr_net = SRNet(self.obs_shape, num_actions, flags, frame_stack_n)
            self.hidden_shape[0] += self.sr_net.hidden_shape[0]
        self.copy_n = self.obs_shape[0] // frame_stack_n

    def normalize(self, x):
        assert x.dtype == torch.uint8
        x = (x.float() - self.norm_low) / \
            (self.norm_high -  self.norm_low)
        return x
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        ch = x.shape[-3]
        x = torch.clamp(x, 0, 1)
        x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x.to(torch.uint8)

    def forward(self, x, actions, one_hot=False, normalize=True, ret_xs=False, ret_zs=False, ret_hs=False):
        """
        Args:
            x(tensor): starting frame (uint if normalize else float) with shape (B, C, H, W)
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
            normalize (tensor): whether to normalize x 
            ret_xs (bool): whther to return predicted states
            ret_zs (bool): whther to return predicted encoding in vp-net
            ret_hs (bool): whther to return model's hidden state
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            done(tensor): predicted done with shape (k, B, ...), in the form of d_{t+1}, d_{t+2}, ..., d_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            logits(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            xs(tensor): predicted states with shape (k+1, B, ...), in the form of x_{t}, x_{t+1}, x_{t+2}, ..., x_{t+k}
            zs(tensor): predicted encoding in vp-net with shape (k+1, B, ...), in the form of z_{t}, z_{t+1}, z_{t+2}, ..., z_{t+k}
            hs(tensor): model's hidden state with shape (k+1, B, ...), in the form of h_{t}, h_{t+1}, h_{t+2}, ..., h_{t+k}
            state(dict): recurrent hidden state with shape (B, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1

        state = {}

        if normalize:
            x = self.normalize(x)
        else:
            assert x.dtype == torch.float

        if self.dual_net:
            sr_net_out = self.sr_net(x, actions, one_hot=one_hot)
            if sr_net_out.xs is not None:
                xs = torch.concat([x.unsqueeze(0), sr_net_out.xs], dim=0)
            else:
                xs = x.unsqueeze(0)
            state.update(sr_net_out.state)
        else:
            sr_net_out = None
            xs = x.unsqueeze(0)
        vp_net_out = self.vp_net(xs, actions, one_hot=one_hot)
        state.update(vp_net_out.state)
        return self._prepare_out(sr_net_out, vp_net_out, ret_hs, ret_xs, ret_zs, state, xs)

    def forward_single(self, state, action, one_hot=False, ret_xs=False, ret_zs=False, ret_hs=False):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            state(dict): recurrent state of the network
            action(tuple): action (int64) with shape (B)
            one_hot (bool): whether to the action use one-hot encoding
        """
        state_ = {}
        if self.dual_net:
            sr_net_out = self.sr_net.forward_single(
                action=action, state=state, one_hot=one_hot
            )
            xs = sr_net_out.xs 
            x = xs[0]
            state_.update(sr_net_out.state)
        else:
            x = None
        vp_net_out = self.vp_net.forward_single(
            action=action, state=state, x=x, one_hot=one_hot
        )
        state_.update(vp_net_out.state)
        return self._prepare_out(sr_net_out, vp_net_out, ret_hs, ret_xs, ret_zs, state_, xs)

    def _prepare_out(self, sr_net_out, vp_net_out, ret_hs, ret_xs, ret_zs, state, xs):
        rd_out = sr_net_out if self.dual_net else vp_net_out
        if ret_hs:
            if self.dual_net:
                hs = torch.concat([sr_net_out.hs, vp_net_out.hs], dim=2)
            else:
                hs = vp_net_out.hs
        else:
            hs = None
        xs = xs if ret_xs else None        
        zs = vp_net_out.pred_zs if ret_zs else None
        return DualNetOut(
            rs=rd_out.rs,
            dones=rd_out.dones,
            vs=vp_net_out.vs,
            logits=vp_net_out.logits,
            xs=xs,
            hs=hs,
            zs=zs,
            state=state,
        )

class RVTran(nn.Module):
    def __init__(self, enc_type, support=300, eps=0.001):
        super(RVTran, self).__init__()
        assert enc_type in [
            1,
            2,
            3,
        ], f"only enc_type [1, 2, 3] is supported, not {enc_type}"
        self.support = support
        self.eps = eps
        self.enc_type = enc_type
        if self.enc_type == 2:
            atom_vector = self.decode_s(torch.arange(-support, support + 1, 1).float())
            self.register_buffer("atom_vector", atom_vector)
            self.encoded_n = 2 * self.support + 1
        elif self.enc_type == 3:
            atom_vector = torch.arange(-support, support + 1, 1)
            self.register_buffer("atom_vector", atom_vector)
            self.encoded_n = 2 * self.support + 1

    def forward(self, x):
        """encode the unencoded scalar reward or values to encoded scalar / vector according to MuZero"""
        with torch.no_grad():
            if self.enc_type == 1:
                enc = self.encode_s(x)
            elif self.enc_type == 2:
                x = torch.clamp(x, self.atom_vector[0], self.atom_vector[-1])
                # Find the indices of the atoms that are greater than or equal to the elements in x
                gt_indices = (self.atom_vector.unsqueeze(0) < x.unsqueeze(-1)).sum(
                    dim=-1
                ) - 1
                gt_indices = torch.clamp(gt_indices, 0, len(self.atom_vector) - 2)

                # Calculate the lower and upper atom bounds for each element in x
                lower_bounds = self.atom_vector[gt_indices]
                upper_bounds = self.atom_vector[gt_indices + 1]

                # Calculate the density between the lower and upper atom bounds
                lower_density = (upper_bounds - x) / (upper_bounds - lower_bounds)
                upper_density = 1 - lower_density

                # Create a zero tensor of shape (3, 4)
                enc = torch.zeros(
                    x.shape + (len(self.atom_vector),),
                    dtype=torch.float32,
                    device=x.device,
                )

                # Use scatter to add the densities to the proper columns
                enc.scatter_(-1, gt_indices.unsqueeze(-1), lower_density.unsqueeze(-1))
                enc.scatter_(
                    -1, (gt_indices + 1).unsqueeze(-1), upper_density.unsqueeze(-1)
                )
            elif self.enc_type == 3:
                enc_s = self.encode_s(x)
                enc_s = torch.clamp(enc_s, -self.support, +self.support)
                enc = torch.zeros(
                    x.shape + (len(self.atom_vector),),
                    dtype=torch.float32,
                    device=x.device,
                )
                enc_floor = torch.floor(enc_s)
                enc_reminder = enc_s - enc_floor
                enc_floor = enc_floor.long().unsqueeze(-1)
                enc.scatter_(
                    -1,
                    torch.clamp_max(self.support + enc_floor + 1, 2 * self.support),
                    enc_reminder.unsqueeze(-1),
                )
                enc.scatter_(
                    -1, self.support + enc_floor, 1 - enc_reminder.unsqueeze(-1)
                )
            return enc

    def encode(self, x):
        return self.forward(x)

    def decode(self, x):
        """decode the encoded vector (or encoded scalar) to unencoded scalar according to MuZero"""
        with torch.no_grad():
            if self.enc_type == 1:
                dec = self.decode_s(x)
            elif self.enc_type == 2:
                dec = torch.sum(self.atom_vector * x, dim=-1)
            elif self.enc_type == 3:
                dec = self.decode_s(torch.sum(self.atom_vector * x, dim=-1))
            return dec

    def encode_s(self, x):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + self.eps * x

    def decode_s(self, x):
        return torch.sign(x) * (
            torch.square(
                (torch.sqrt(1 + 4 * self.eps * (torch.abs(x) + 1 + self.eps)) - 1)
                / (2 * self.eps)
            )
            - 1
        )
