from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from thinker import util
from thinker.core.rnn import ConvAttnLSTM

ActorOut = namedtuple(
    "ActorOut",
    [
        "policy_logits",
        "im_policy_logits",
        "reset_policy_logits",
        "action",
        "im_action",
        "reset_action",
        "baseline",
        "baseline_enc",
        "reg_loss",
    ],
)
OutNetOut = namedtuple(
    "OutNetOut",
    [
        "single_rs",
        "rs",
        "r_enc_logits",
        "dones",
        "done_logits",
        "vs",
        "v_enc_logits",
        "logits",
        "state",
    ],
)
ModelNetOut = namedtuple(
    "ModelNetOut",
    ["single_rs", "rs", "r_enc_logits", "dones", "done_logits", "xs", "hs", "state"],
)
PredNetOut = namedtuple(
    "PredNetOut",
    [
        "single_rs",
        "rs",
        "r_enc_logits",
        "dones",
        "done_logits",
        "vs",
        "v_enc_logits",
        "logits",
        "hs",
        "pred_zs",
        "true_zs",
        "state",
    ],
)
DuelNetOut = namedtuple(
    "DuelNetOut", ["single_rs", "rs", "dones", "vs", "logits", "ys", "zs", "state"]
)


def add_hw(x, h, w):
    return x.unsqueeze(-1).unsqueeze(-1).broadcast_to(x.shape + (h, w))


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


class ActorEncoder(nn.Module):
    def __init__(self, input_shape, num_actions, flags):
        super(ActorEncoder, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.flags = flags
        self.frame_encode = flags.actor_see_type == 0
        self.out_size = 256

        if flags.actor_see_type == 0:
            # see the frame directly; we need to have frame encoding
            self.frame_encoder = FrameEncoder(
                input_shape=input_shape,
                num_actions=num_actions,
                downscale_c=2,
                size_nn=flags.model_size_nn,
                concat_action=False,
                grayscale=flags.grayscale,
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
        if self.flags.actor_see_type == 0:
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


class ActorNetBase(BaseNet):
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):
        super(ActorNetBase, self).__init__()
        self.obs_shape = obs_shape
        self.gym_obs_shape = gym_obs_shape
        self.num_actions = num_actions

        self.tran_t = flags.tran_t  # number of recurrence of RNN
        self.tran_mem_n = flags.tran_mem_n  # size of memory for the attn modules
        self.tran_layer_n = flags.tran_layer_n  # number of layers
        self.tran_lstm_no_attn = (
            flags.tran_lstm_no_attn
        )  # to use attention in lstm or not
        self.attn_mask_b = flags.tran_attn_b  # atention bias for current position
        self.tran_dim = flags.tran_dim  # size of transformer / LSTM embedding dim
        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.actor_see_type = (
            flags.actor_see_type
        )  # -1 for nothing, 0. for predicted / true frame, 1. for z, 2. for h.
        self.disable_model = flags.disable_model
        self.disable_mem = flags.disable_mem
        self.disable_rnn = flags.disable_rnn
        self.flags = flags

        # encoder for state or encoding output
        last_out_size = self.num_actions + self.num_rewards
        if not self.disable_model:
            last_out_size += self.num_actions + 2

        if self.actor_see_type >= 0:
            if self.actor_see_type == 0:
                input_shape = gym_obs_shape
            else:
                in_channels = int(256 // flags.model_downscale_c)
                if self.actor_see_type == 3:
                    in_channels *= 2
                input_shape = (
                    in_channels,
                    gym_obs_shape[1] // 16 + int((gym_obs_shape[1] % 16) > 0),
                    gym_obs_shape[2] // 16 + int((gym_obs_shape[2] % 16) > 0),
                )
            self.actor_encoder = ActorEncoder(
                input_shape=input_shape, num_actions=num_actions, flags=flags
            )
            last_out_size += self.actor_encoder.out_size

        if self.obs_shape is not None:
            # there is model's tree stat. output
            if not self.disable_rnn:
                self.initial_enc = nn.Sequential(
                    nn.Linear(self.obs_shape[0], self.tran_dim), nn.ReLU()
                )
                if self.tran_layer_n >= 1:
                    self.core = ConvAttnLSTM(
                        h=1,
                        w=1,
                        input_dim=self.tran_dim,
                        hidden_dim=self.tran_dim,
                        kernel_size=1,
                        num_layers=self.tran_layer_n,
                        num_heads=8,
                        mem_n=self.tran_mem_n,
                        attn=not self.tran_lstm_no_attn,
                        attn_mask_b=self.attn_mask_b,
                    )
                self.model_stat_fc = nn.Sequential(
                    nn.Linear(self.tran_dim, self.tran_dim), nn.ReLU()
                )
                last_out_size += self.tran_dim
            else:
                # self.core = mlp(input_size=self.obs_shape[0], layer_sizes=[400, 400, 400], output_size=100, norm=False)
                self.core = MLPWithSkipConnections(
                    input_size=self.obs_shape[0],
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                last_out_size += 100

        self.policy = nn.Linear(last_out_size, self.num_actions)

        if not self.disable_model:
            self.im_policy = nn.Linear(last_out_size, self.num_actions)
            self.reset = nn.Linear(last_out_size, 2)

        self.rv_tran = None
        self.baseline = nn.Linear(last_out_size, self.num_rewards)
        if self.flags.reward_clipping > 0:
            self.baseline_clamp = self.flags.reward_clipping / (
                1 - self.flags.discounting
            )

        if flags.critic_zero_init:
            nn.init.constant_(self.baseline.weight, 0.0)
            nn.init.constant_(self.baseline.bias, 0.0)

    def initial_state(self, batch_size, device=None):
        if (
            self.obs_shape is not None
            and self.tran_layer_n >= 1
            and not self.disable_rnn
        ):
            state = self.core.init_state(batch_size, device=device)
        else:
            state = ()
        return state

    def forward(self, obs, core_state=(), greedy=False):
        """one-step forward for the actor;
        args:
            obs (EnvOut):
                model_out (tensor): model statistic output with shape (T x B x C)
                model_encodes (tensor): model encoding output with shape (T x B x C X H X W)
                gym_env_out (tensor): frame output (only for perfect_model) with shape (T x B x C X H X W)
                done  (tensor): if episode ends with shape (T x B)
                cur_t (tensor): current planning step with shape (T x B)
                and other environment output that is not used.
        return:
            ActorOut:
                policy_logits (tensor): logits of real action (T x B x |A|)
                im_policy_logits (tensor): logits of imagine action (T x B x |A|)
                reset_policy_logits (tensor): logits of real action (T x B x 2)
                action (tensor): sampled real action (non-one-hot form) (T x B)
                im_action (tensor): sampled imagine action (non-one-hot form) (T x B)
                reset_action (tensor): sampled reset action (non-one-hot form) (T x B)
                baseline (tensor): prediced baseline (T x B x 1) or (T x B x 2)
                reg_loss (tensor): regularization loss (T x B)
        """
        done = obs.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape

        final_out = []

        last_action = torch.flatten(obs.last_action, 0, 1)
        last_re_action = F.one_hot(last_action[:, 0], self.num_actions)
        final_out.append(last_re_action)

        if not self.disable_model:
            last_im_action = F.one_hot(last_action[:, 1], self.num_actions)
            last_reset_action = F.one_hot(last_action[:, 2], 2)
            final_out.append(last_im_action)
            final_out.append(last_reset_action)

        last_reward = torch.clamp(torch.flatten(obs.reward, 0, 1), -1, +1)
        final_out.append(last_reward)

        # compute model encoding branch
        if self.actor_see_type >= 0:
            if self.actor_see_type == 0 and (
                self.disable_model or self.flags.perfect_model
            ):
                model_enc = obs.gym_env_out.float() / 255.0
            else:
                model_enc = obs.model_encodes
            model_enc = torch.flatten(model_enc, 0, 1)
            model_enc = self.actor_encoder(model_enc)
            final_out.append(model_enc)

        # compute model stat branch
        if self.obs_shape is not None:
            if not self.disable_rnn:
                model_stat = torch.flatten(obs.model_out, 0, 1)
                model_stat = self.initial_enc(model_stat)
                if self.tran_layer_n >= 1:
                    core_input = model_stat.view(*((T, B) + model_stat.shape[1:]))
                    core_output_list = []
                    notdone = ~(done.bool())
                    core_input = core_input.unsqueeze(-1).unsqueeze(-1)
                    for n, (input, nd) in enumerate(
                        zip(core_input.unbind(), notdone.unbind())
                    ):
                        if self.disable_mem:
                            core_state = tuple(torch.zeros_like(s) for s in core_state)
                        for t in range(self.tran_t):
                            if t > 0:
                                nd = torch.ones_like(nd)
                            nd = nd.view(-1)
                            output, core_state = self.core(
                                input, core_state, nd, nd
                            )  # output shape: 1, B, core_output_size
                            # output, core_state = checkpoint(self.core, input, core_state, nd, nd)
                        core_output_list.append(output)
                    core_output = torch.cat(core_output_list)
                    core_output = torch.flatten(core_output, 0, 1)
                    model_stat = torch.flatten(core_output, start_dim=1)
                model_stat = self.model_stat_fc(model_stat)
                final_out.append(model_stat)
            else:
                model_stat = torch.flatten(obs.model_out, 0, 1)
                model_stat = self.core(model_stat)
                final_out.append(model_stat)

        final_out = torch.concat(final_out, dim=-1)
        policy_logits = self.policy(final_out)
        if not greedy:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        action = action.view(T, B)

        if not self.disable_model:
            im_policy_logits = self.im_policy(final_out)
            if not greedy:
                im_action = torch.multinomial(
                    F.softmax(im_policy_logits, dim=1), num_samples=1
                )
            else:
                im_action = torch.argmax(im_policy_logits, dim=1)
            im_policy_logits = im_policy_logits.view(T, B, self.num_actions)
            im_action = im_action.view(T, B)
            reset_policy_logits = self.reset(final_out)
            if not greedy:
                reset_action = torch.multinomial(
                    F.softmax(reset_policy_logits, dim=1), num_samples=1
                )
            else:
                reset_action = torch.argmax(reset_policy_logits, dim=1)
            reset_policy_logits = reset_policy_logits.view(T, B, 2)
            reset_action = reset_action.view(T, B)
        else:
            im_policy_logits, im_action, reset_policy_logits, reset_action = (
                None,
                None,
                None,
                None,
            )

        baseline = self.baseline(final_out)
        if self.flags.reward_clipping > 0:
            baseline = torch.clamp(
                baseline, -self.baseline_clamp, +self.baseline_clamp
            )
        baseline_enc = None

        baseline_enc = (
            baseline_enc.view((T, B) + baseline_enc.shape[1:])
            if baseline_enc is not None
            else None
        )
        baseline = baseline.view(T, B, self.num_rewards)

        reg_loss = (
            1e-3 * torch.sum(policy_logits**2, dim=-1) / 2
            + 1e-6 * torch.sum(final_out**2, dim=-1).view(T, B) / 2
        )
        if not self.disable_model:
            reg_loss += (
                1e-3 * torch.sum(im_policy_logits**2, dim=-1) / 2
                + 1e-3 * torch.sum(reset_policy_logits**2, dim=-1) / 2
            )

        actor_out = ActorOut(
            policy_logits=policy_logits,
            im_policy_logits=im_policy_logits,
            reset_policy_logits=reset_policy_logits,
            action=action,
            im_action=im_action,
            reset_action=reset_action,
            baseline_enc=baseline_enc,
            baseline=baseline,
            reg_loss=reg_loss,
        )

        return actor_out, core_state


class DRCNet(BaseNet):
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):
        super(DRCNet, self).__init__()
        assert flags.disable_model

        self.obs_shape = obs_shape
        self.gym_obs_shape = gym_obs_shape
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
        )
        output_shape = lambda h, w, kernel, stride, padding: (
            ((h + 2 * padding - kernel) // stride + 1),
            ((w + 2 * padding - kernel) // stride + 1),
        )

        h, w = output_shape(gym_obs_shape[1], gym_obs_shape[2], 8, 4, 2)
        h, w = output_shape(h, w, 4, 2, 1)

        self.drc_depth = 3
        self.drc_n = 3
        self.core = ConvAttnLSTM(
            h=h,
            w=w,
            input_dim=32,
            hidden_dim=32,
            kernel_size=3,
            num_layers=3,
            num_heads=8,
            mem_n=None,
            attn=False,
            attn_mask_b=None,
            pool_inject=True,
        )
        last_out_size = 32 * h * w * 2
        self.final_layer = nn.Linear(last_out_size, 256)
        self.policy = nn.Linear(256, self.num_actions)
        self.baseline = nn.Linear(256, 1)

    def initial_state(self, batch_size, device=None):
        return self.core.init_state(batch_size, device=device)

    def forward(self, obs, core_state=(), greedy=False):
        done = obs.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape
        model_enc = obs.gym_env_out.float() / 255.0
        model_enc = torch.flatten(model_enc, 0, 1)
        model_enc = self.encoder(model_enc)
        core_input = model_enc.view(*((T, B) + model_enc.shape[1:]))
        core_output_list = []
        notdone = ~(done.bool())
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):
            for t in range(self.drc_n):
                if t > 0:
                    nd = torch.ones_like(nd)
                nd = nd.view(-1)
                output, core_state = self.core(input, core_state, nd, nd)
            core_output_list.append(output)
        core_output = torch.cat(core_output_list)
        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.cat([model_enc, core_output], dim=1)
        core_output = torch.flatten(core_output, 1)
        final_out = F.relu(self.final_layer(core_output))
        policy_logits = self.policy(final_out)
        if not greedy:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        action = action.view(T, B)
        baseline = self.baseline(final_out).view(T, B, 1)
        reg_loss = (
            1e-3 * torch.sum(policy_logits**2, dim=-1)
            + 1e-5 * torch.sum(torch.square(self.baseline.weight))
            + 1e-5 * torch.sum(torch.square(self.policy.weight))
        )
        actor_out = ActorOut(
            policy_logits=policy_logits,
            im_policy_logits=None,
            reset_policy_logits=None,
            action=action,
            im_action=None,
            reset_action=None,
            baseline_enc=None,
            baseline=baseline,
            reg_loss=reg_loss,
        )
        return actor_out, core_state


def ActorNet(obs_shape, gym_obs_shape, num_actions, flags):
    if flags.drc:
        return DRCNet(obs_shape, gym_obs_shape, num_actions, flags)
    else:
        return ActorNetBase(obs_shape, gym_obs_shape, num_actions, flags)

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
        # x = checkpoint_sequential(self.d_conv, segments=1, input=z)
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


class Output_rvpi(nn.Module):
    def __init__(
        self,
        num_actions,
        input_shape,
        value_clamp,
        max_unroll_length,
        zero_init,
        size_nn,
        predict_v_pi=True,
        predict_r=True,
        predict_done=False,
        disable_bn=False,
        prefix="",
    ):
        super(Output_rvpi, self).__init__()

        self.input_shape = input_shape
        self.size_nn = size_nn
        self.value_clamp = value_clamp
        self.max_unroll_length = max_unroll_length
        self.predict_v_pi = predict_v_pi
        self.predict_r = predict_r
        self.predict_done = predict_done
        self.prefix = prefix


        c, h, w = input_shape
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
            r_enc_logit = None
            r = r_out.squeeze(-1)
            single_r = None
        else:
            single_r, r, r_enc_logit = None, None, None
        out = OutNetOut(
            single_rs=single_r,
            rs=r,
            r_enc_logits=r_enc_logit,
            dones=done,
            done_logits=done_logit,
            vs=v,
            v_enc_logits=v_enc_logit,
            logits=logits,
            state=state_,
        )
        return out

    def init_state(self, bsz, device):
        return {}


class ModelNetV(nn.Module):
    def __init__(self, obs_shape, num_actions, flags):
        super(ModelNetV, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.size_nn = (
            flags.model_size_nn
        )  # size_nn: int to adjust for the depth of model net
        self.downscale_c = (
            flags.model_downscale_c
        )  # downscale_c: int to downscale number of channels; default=2
        self.frame_copy = flags.frame_copy
        self.grayscale = flags.grayscale
        self.encoder = FrameEncoder(
            num_actions=num_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=True,
            frame_copy=self.frame_copy,
            grayscale=self.grayscale,
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
        if self.flags.reward_clipping > 0:
            value_clamp = self.flags.reward_clipping / (1 - self.flags.discounting)
        else:
            value_clamp = None
        self.out = Output_rvpi(
            num_actions=num_actions,
            input_shape=self.hidden_shape,
            value_clamp=value_clamp,
            max_unroll_length=flags.model_k_step_return,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=False,
            predict_r=True,
            predict_done=self.flags.model_done_loss_cost > 0.0,
            disable_bn=self.flags.model_disable_bn,
            prefix="m_",
        )
        self.rv_tran = self.out.rv_tran

    def forward(self, x, actions, one_hot=False):
        """
        Args:
            x(tensor): frames (float) with shape (B, C, H, W), in the form of s_t
            actions(tensor): action (int64) with shape (k+1, B, *), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            ModelNetOut tuple with predicted rewards (rs), images (xs), done (dones) in the shape of (k, B, ...);
                in the form of y_{t+1}, y_{t+2}, ..., y_{t+k} and states with element in the shape of (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)
        if k > 1:
            util.print_mem("M1.1.1")
        h = self.encoder(x, actions[0])
        hs = [h.unsqueeze(0)]
        if k > 1:
            util.print_mem("M1.1.2")
        for t in range(1, k + 1):
            h = self.RNN(h=h, actions=actions[t])
            hs.append(h.unsqueeze(0))
            if k > 1:
                util.print_mem("M1.1.3")
        hs = torch.concat(hs, dim=0)
        if k > 1:
            util.print_mem("M1.1.4")

        state = {"m_h": h}
        if len(hs) > 1:
            xs = self.encoder.decode(hs[1:], flatten=True)
            if k > 1:
                util.print_mem("M1.1.5")
            if self.frame_copy:
                copy_n = 1 if self.grayscale else 3
                stacked_x = x
                stacked_xs = []
                for i in range(k):
                    stacked_x = torch.concat([stacked_x[:, copy_n:], xs[i]], dim=1)
                    stacked_xs.append(stacked_x)
                xs = torch.stack(stacked_xs, dim=0)
                state["last_x"] = stacked_x[:, copy_n:]
            if k > 1:
                util.print_mem("M1.1.6")
        else:
            xs = None
            if self.frame_copy:
                copy_n = 1 if self.grayscale else 3
                state["last_x"] = x[:, copy_n:]
        if k > 1:
            util.print_mem("M1.1.7")
        outs = []
        r_state = self.out.init_state(bsz=b, device=x.device)
        for t in range(1, k + 1):
            out = self.out(hs[t], predict_reward=True, state=r_state)
            outs.append(out)
            r_state = out.state
        if k > 1:
            util.print_mem("M1.1.8")
        state.update(r_state)
        return ModelNetOut(
            single_rs=util.safe_concat(outs, "single_rs", 0),
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
        h = self.RNN(h=state["m_h"], actions=action)
        x = self.encoder.decode(h, flatten=False)
        if self.frame_copy:
            x = torch.concat([state["last_x"], x], dim=1)

        out = self.out(h, predict_reward=True, state=state)
        state = {"m_h": h}
        state.update(out.state)
        if self.frame_copy:
            state["last_x"] = x[:, (1 if self.grayscale else 3) :]

        return ModelNetOut(
            single_rs=util.safe_unsqueeze(out.single_rs, 0),
            rs=util.safe_unsqueeze(out.rs, 0),
            r_enc_logits=util.safe_unsqueeze(out.r_enc_logits, 0),
            dones=util.safe_unsqueeze(out.dones, 0),
            done_logits=util.safe_unsqueeze(out.done_logits, 0),
            xs=util.safe_unsqueeze(x, 0),
            hs=util.safe_unsqueeze(h, 0),
            state=state,
        )


class PredNetV(nn.Module):
    def __init__(self, obs_shape, num_actions, flags):
        super(PredNetV, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.size_nn = (
            flags.model_size_nn
        )  # size_nn: int to adjust for size of model net
        self.downscale_c = (
            flags.model_downscale_c
        )  # downscale_c: int to downscale number of channels; default=2
        self.use_rnn = (
            not flags.perfect_model
        )  # dont use rnn if we have perfect dynamic
        self.receive_z = (
            flags.duel_net
        )  # rnn receives z only when we are using duel net
        self.predict_rd = (
            not flags.duel_net and not flags.perfect_model
        )  # network also predicts reward and done if not duel net under non-perfect dynamic
        self.grayscale = flags.grayscale

        self.encoder = FrameEncoder(
            num_actions=num_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=False,
            grayscale=self.grayscale,
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
        if self.flags.reward_clipping > 0:
            value_clamp = self.flags.reward_clipping / (1 - self.flags.discounting)
        else:
            value_clamp = None
        self.out = Output_rvpi(
            num_actions=num_actions,
            input_shape=self.hidden_shape,
            value_clamp=value_clamp,
            max_unroll_length=flags.model_k_step_return,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=True,
            predict_r=self.predict_rd,
            predict_done=self.predict_rd and self.flags.model_done_loss_cost > 0.0,
            disable_bn=self.flags.model_disable_bn,
            prefix="p_",
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
            PredNetOut tuple with predicted values (vs), policies (logits) in the shape of (k+1, B, ...);
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
        r_state = self.out.init_state(bsz=b, device=xs.device)
        for t in range(0, k + 1):
            out = self.out(hs[t], predict_reward=t > 0, state=r_state)
            outs.append(out)
            r_state = out.state

        if not self.receive_z:
            pred_zs = torch.concat([zs[[0]], self.h_to_z(hs[1:], flatten=True)], dim=0)
        else:
            pred_zs = zs

        state = {"p_h": h}
        state.update(r_state)
        return PredNetOut(
            single_rs=util.safe_concat(outs[1:], "single_rs", 0),
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
            rnn_in = torch.concat([state["p_h"], z], dim=1)
        else:
            rnn_in = state["p_h"]
        h = self.RNN(h=rnn_in, actions=action)
        out = self.out(h, predict_reward=True, state=state)
        state = {"p_h": h}
        state.update(out.state)

        if not self.receive_z:
            pred_z = self.h_to_z(h, flatten=False)
        else:
            pred_z = z

        return PredNetOut(
            single_rs=util.safe_unsqueeze(out.single_rs, 0),
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


class DuelNetBase(BaseNet):
    def __init__(self, obs_shape, num_actions, flags, debug=False):
        super(DuelNetBase, self).__init__()
        self.rnn = False
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.size_nn = (
            flags.model_size_nn
        )  # size_nn: int to adjust for size of model net
        self.duel_net = flags.duel_net
        if self.duel_net:
            self.model_net = ModelNetV(obs_shape, num_actions, flags)
        self.pred_net = PredNetV(obs_shape, num_actions, flags)
        self.debug = debug

    def forward(self, x, actions, one_hot=False, rescale=True, ret_zs=False):
        """
        Args:
            x(tensor): starting frame (uint if rescale else float) with shape (B, C, H, W)
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            done(tensor): predicted done with shape (k, B, ...), in the form of d_{t+1}, d_{t+2}, ..., d_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            logits(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            ys(tensor): output to actor with shape (k+1, B, ...), in the form of y_{t}, y_{t+1}, y_{t+2}, ..., y_{t+k}
            state(dict): recurrent hidden state with shape (B, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1

        state = {}

        if rescale:
            assert x.dtype == torch.uint8
            x = x.float() / 255.0
        else:
            assert x.dtype == torch.float

        if self.duel_net:
            model_net_out = self.model_net(x, actions, one_hot=one_hot)
            if model_net_out.xs is not None:
                xs = torch.concat([x.unsqueeze(0), model_net_out.xs], dim=0)
            else:
                xs = x.unsqueeze(0)
            state.update(model_net_out.state)
        else:
            xs = x.unsqueeze(0)
        pred_net_out = self.pred_net(xs, actions, one_hot=one_hot)
        state.update(pred_net_out.state)

        if self.flags.actor_see_type == 0:
            ys = xs
        elif self.flags.actor_see_type == 1:
            ys = pred_net_out.pred_zs
        elif self.flags.actor_see_type == 2:
            ys = pred_net_out.hs
        elif self.flags.actor_see_type == 3:
            ys = torch.concat([model_net_out.hs, pred_net_out.hs], dim=2)
        else:
            ys = None

        rd_out = model_net_out if self.duel_net else pred_net_out

        if ret_zs:
            zs = pred_net_out.pred_zs
        else:
            zs = None

        if self.debug:
            state["pred_xs"] = xs[-1]

        return DuelNetOut(
            single_rs=rd_out.single_rs,
            rs=rd_out.rs,
            dones=rd_out.dones,
            vs=pred_net_out.vs,
            logits=pred_net_out.logits,
            ys=ys,
            zs=zs,
            state=state,
        )

    def forward_single(self, state, action, one_hot=False, ret_zs=False):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            state(dict): recurrent state of the network
            action(tuple): action (int64) with shape (B)
            one_hot (bool): whether to the action use one-hot encoding
        """
        state_ = {}
        if self.duel_net:
            model_net_out = self.model_net.forward_single(
                action=action, state=state, one_hot=one_hot
            )
            x = model_net_out.xs[0]
            state_.update(model_net_out.state)
        else:
            x = None
        pred_net_out = self.pred_net.forward_single(
            action=action, state=state, x=x, one_hot=one_hot
        )
        state_.update(pred_net_out.state)

        if self.flags.actor_see_type == 0:
            ys = util.safe_unsqueeze(x, 0)
        elif self.flags.actor_see_type == 1:
            ys = pred_net_out.pred_zs
        elif self.flags.actor_see_type == 2:
            ys = pred_net_out.hs
        elif self.flags.actor_see_type == 3:
            ys = torch.concat([model_net_out.hs, pred_net_out.hs], dim=2)
        else:
            ys = None

        if ret_zs:
            zs = pred_net_out.pred_zs
        else:
            zs = None

        rd_out = model_net_out if self.duel_net else pred_net_out
        if self.debug:
            state_["pred_xs"] = x if x is not None else None
        return DuelNetOut(
            single_rs=rd_out.single_rs,
            rs=rd_out.rs,
            dones=rd_out.dones,
            vs=pred_net_out.vs,
            logits=pred_net_out.logits,
            ys=ys,
            zs=zs,
            state=state_,
        )


def ModelNet(obs_shape, num_actions, flags, debug=False):
    return DuelNetBase(obs_shape, num_actions, flags, debug)

