from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker import util
from thinker.core.module import conv3x3, ResBlock, MLP, OneDResBlock
from thinker.core.rnn import ConvAttnLSTM, LSTMReset
import math

OutNetOut = namedtuple(
    "OutNetOut",
    [
        "rs", "r_enc_logits", "dones", "done_logits", "vs", "v_enc_logits", "policy"
    ],
)
SRNetOut = namedtuple(
    "SRNetOut",
    ["rs", "r_enc_logits", "dones", "done_logits", "xs", "hs", "state", "noise_loss"],
)
VPNetOut = namedtuple(
    "VPNetOut",
    ["rs", "r_enc_logits", "dones", "done_logits", "vs", "v_enc_logits",
        "policy", "hs", "pred_zs", "true_zs", "state",
    ],
)
DualNetOut = namedtuple(
    "DualNetOut", ["rs", "dones", "vs", "v_enc_logits", "policy", "xs", "hs", "zs", "state"]
)

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
        prefix,
        dim_rep_actions,
        input_shape,
        size_nn=1,
        downscale_c=2,
        downscale=True,
        concat_action=True,
        decoder=False,
        decoder_depth=0,
        frame_stack_n=1,
        disable_bn=False,
        has_memory=False,
    ):
        super(FrameEncoder, self).__init__()
        self.prefix = prefix
        self.dim_rep_actions = dim_rep_actions
        self.size_nn = size_nn
        self.downscale_c = downscale_c
        self.decoder = decoder
        self.decoder_depth = decoder_depth
        self.frame_stack_n = frame_stack_n        
        self.input_shape = input_shape        
        self.concat_action = concat_action
        self.oned_input = len(self.input_shape) == 1
        self.has_memory = has_memory

        stride = 2 if downscale else 1
        frame_channels = input_shape[0]
        if self.concat_action:
            in_channels = frame_channels + dim_rep_actions
        else:
            in_channels = frame_channels

        if not self.oned_input:
            h, w = input_shape[1:]
            frame_channels, h, w = input_shape
            if self.concat_action:
                in_channels = frame_channels + dim_rep_actions
            else:
                in_channels = frame_channels

            n_block = 1 * self.size_nn
            out_channels = int(128 // downscale_c)

            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
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
                stride=stride,
                padding=1,
            )
            res = [
                ResBlock(inplanes=out_channels * 2, disable_bn=disable_bn)
                for _ in range(n_block)
            ]  # Deep: 3 blocks here
            self.res2 = nn.Sequential(*res)
            self.avg1 = nn.AvgPool2d(3, stride=stride, padding=1)
            res = [
                ResBlock(inplanes=out_channels * 2, disable_bn=disable_bn)
                for _ in range(n_block)
            ]  # Deep: 3 blocks here
            self.res3 = nn.Sequential(*res)
            self.avg2 = nn.AvgPool2d(3, stride=stride, padding=1)
            self.out_shape = (
                out_channels * 2,
                h // 16 + int((h % 16) > 0) if stride == 2 else h,
                w // 16 + int((w % 16) > 0) if stride == 2 else w,
            )

            if self.has_memory:
                raise NotImplementedError()

            if self.decoder:
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
                for i in range(4-self.decoder_depth):
                    if i in [1, 3]:
                        d_conv.extend(
                            [ResBlock(inplanes=conv_channels[4 - i], disable_bn=disable_bn) for _ in range(n_block)]
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
        else:
            n_block = 2 * size_nn
            hidden_size = 512 // downscale_c
            self.hidden_size = hidden_size
            self.input_block = nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Tanh()
            )            
            self.res = nn.Sequential(*[OneDResBlock(hidden_size) for _ in range(n_block)])    
            self.out_shape = (hidden_size,)
            
            if self.has_memory:
                #self.rnn = ConvAttnLSTM(input_dim=hidden_size, hidden_dim=hidden_size//2, num_layers=2, attn=False)
                self.rnn = LSTMReset(input_dim=hidden_size, hidden_dim=hidden_size, num_layers=2)
                #self.rnn_fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())   

            if self.decoder:
                self.d_res = nn.Sequential(*[OneDResBlock(hidden_size) for _ in range(n_block)])
                self.output_block = nn.Linear(hidden_size, input_shape[0])            

        self.initial_state(batch_size=1)

    def initial_state(self, batch_size=1, device=None):
        state = {}
        self.per_state_len = 0    
        if self.has_memory:
            enc_state = self.rnn.initial_state(bsz=batch_size, device=device)
            self.per_state_len = len(enc_state)
            for i in range(self.per_state_len): state[f"per_{self.prefix}_{i}"] = enc_state[i]
            #state[f"per_{self.prefix}_dbg_state"] = torch.zeros((batch_size, ) + self.input_shape, device=device)
            #state[f"per_{self.prefix}_dbg_action"] = torch.zeros(batch_size, self.actions_ch, device=device)
        return state
    
    def forward_pre_mem(self, x, actions=None, flatten=False, depth=0, end_depth=None):
        """
        Args:
          x (tensor): frame with shape T, B, * or B, *
          action (tensor): action with shape B, dim_rep_actions (encoded action)
          flatten (bool): whether the input is in shape (T, B, *) or (B, *); the return shape will be in same form as input shape
          depth (int): starting depth - 0 means foward pass from first layer
          end_depth (int): depth from top to return - 0 means foward pass till last layer
        """
        assert x.dtype in [torch.float, torch.float16]
        if self.concat_action: assert actions.dtype in [torch.float, torch.float16]
        if self.oned_input and flatten:
            assert len(x.shape) == 3, f"x should have 3 dim instead of shape {x.shape}"
        elif self.oned_input and not flatten:
            assert len(x.shape) == 2, f"x should have 2 dim instead of shape {x.shape}"
        elif not self.oned_input and flatten:
            assert len(x.shape) == 5, f"x should have 5 dim instead of shape {x.shape}"
        elif not self.oned_input and not flatten:
            assert len(x.shape) == 4, f"x should have 4 dim instead of shape {x.shape}"

        input_shape = x.shape

        if end_depth is not None and end_depth <= 0: return x
        if flatten:            
            x = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])
        if self.concat_action and depth <= 0:
            if flatten:
                actions = actions.view(
                    (actions.shape[0] * actions.shape[1],) + actions.shape[2:]
                )
            if not self.oned_input:
                actions = (
                    actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])
                )                
            x = torch.concat([x, actions], dim=1)

        if not self.oned_input:
            if depth <= 0:
                x = F.relu(self.conv1(x))
                x = self.res1(x)
            if end_depth is not None and end_depth <= 1: return self.post_process(x, input_shape, flatten)
            if depth <= 1:
                x = F.relu(self.conv2(x))
                x = self.res2(x)
            if end_depth is not None and end_depth <= 2: return self.post_process(x, input_shape, flatten)
            if depth <= 2:
                x = self.avg1(x)
                x = self.res3(x)
            if end_depth is not None and end_depth <= 3: return self.post_process(x, input_shape, flatten)
            if depth <= 3:
                x = self.avg2(x)            
        else:
            assert depth == 0
            x = self.input_block(x)
            x = self.res(x)
        return self.post_process(x, input_shape, flatten)
    
    def post_process(self, x, input_shape, flatten):
        if flatten:
            x = x.view(input_shape[:2] + x.shape[1:])
        return x

    def forward(self, x, done, actions, state={}, flatten=False, depth=0):        
        new_state = {}
        x = self.forward_pre_mem(x=x, actions=actions, flatten=flatten, depth=depth)        
        if self.has_memory:
            if not self.oned_input:
                raise NotImplementedError()
            else:             
                enc_state = tuple(state[f"per_{self.prefix}_{i}"] for i in range(self.per_state_len))    
                if not flatten: 
                    x = x.unsqueeze(0)
                    if done is not None: done = done.unsqueeze(0)
                if done is None: done = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)                        
                x, enc_state = self.rnn(x, done, enc_state)                
                if not flatten: x = x.squeeze(0)
                for i in range(self.per_state_len): new_state[f"per_{self.prefix}_{i}"] = enc_state[i]
        return x, new_state

    def decode(self, z, flatten=False):
        """
        Args:
          z (tensor): encoding with shape B, *
        """
        if flatten:
            input_shape = z.shape
            z = z.view((z.shape[0] * z.shape[1],) + z.shape[2:])
        if not self.oned_input:
            x = self.d_conv(z)
            decoded_h = self.input_shape[1] // (2 ** self.decoder_depth)
            decoded_w = self.input_shape[2] // (2 ** self.decoder_depth)
            if x.shape[2] > decoded_h: x = x[:, :, :decoded_h]
            if x.shape[3] > decoded_w: x = x[:, :, :, :decoded_w]
        else:
            x = self.d_res(z)
            x = self.output_block(x)
        if flatten:
            x = x.view(input_shape[:2] + x.shape[1:])        
        return x

class DynamicModel(nn.Module):
    def __init__(
        self,
        dim_rep_actions,
        inplanes,
        oned_input,
        size_nn=1,
        outplanes=None,
        disable_half_grad=True,
        disable_bn=False,
    ):
        super(DynamicModel, self).__init__()
        self.dim_rep_actions = dim_rep_actions
        self.inplanes = inplanes
        self.size_nn = size_nn
        self.disable_half_grad = disable_half_grad
        if outplanes is None:
            outplanes = inplanes
        self.oned_input = oned_input

        if not self.oned_input:
            res = [
                ResBlock(
                    inplanes=inplanes + dim_rep_actions,
                    outplanes=outplanes,
                    disable_bn=disable_bn,
                )
            ] + [
                ResBlock(inplanes=outplanes, disable_bn=disable_bn)
                for i in range(4 * self.size_nn)
            ]
            self.res = nn.Sequential(*res)
        else:
            n_block = 2 * size_nn
            self.input_block = nn.Sequential(
                nn.Linear(inplanes + dim_rep_actions, outplanes),
                nn.LayerNorm(outplanes),
                nn.ReLU()
            )
            self.res = nn.Sequential(*[OneDResBlock(outplanes) for _ in range(n_block)])
        self.outplanes = outplanes

    def forward(self, h, actions):
        x = h
        if self.training and not self.disable_half_grad:
            # no half-gradient for dreamer net
            x.register_hook(lambda grad: grad * 0.5)
        if not self.oned_input:
            actions = (
                actions.unsqueeze(-1).unsqueeze(-1).tile([1, 1, x.shape[2], x.shape[3]])
            )
            x = torch.concat([x, actions], dim=1)
            out = self.res(x)
        else:
            x = torch.concat([x, actions], dim=1)
            x = self.input_block(x)
            out = self.res(x)
        return out

class NoiseModel(nn.Module):
    def __init__(
        self,
        in_shape,
        size_nn,
        noise_n=20,
        noise_d=10,
    ):
        super(NoiseModel, self).__init__()
        self.in_shape = in_shape
        self.size_nn = size_nn
        self.noise_n = noise_n
        self.noise_d = noise_d

        assert len(self.in_shape) == 3

        res = [
            ResBlock(
                inplanes=in_shape[0],
                outplanes=32,
                disable_bn=True,
            )
        ] + [
            ResBlock(inplanes=32, disable_bn=True)
            for i in range(4 * self.size_nn)
        ]
        self.res = nn.Sequential(*res)
        self.fc = nn.Linear(32*in_shape[1]*in_shape[2], noise_n*noise_d)
        self.out_shape = (noise_n, noise_d)

    def forward(self, x):
        out = self.res(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out).view(out.shape[0], self.noise_n, self.noise_d)
        return out

class OutputNet(nn.Module):
    def __init__(
        self,
        action_space,
        input_shape,
        enc_type,
        enc_f_type,
        zero_init,
        size_nn,
        predict_v_pi=True,
        predict_r=True,
        predict_done=False,
        ordinal=False,
    ):
        super(OutputNet, self).__init__()

        self.action_space = action_space
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)
        self.input_shape = input_shape        
        self.oned_input = len(self.input_shape) == 1
        self.size_nn = size_nn
        self.enc_type = enc_type
        self.predict_v_pi = predict_v_pi
        self.predict_r = predict_r
        self.predict_done = predict_done  
        self.ordinal = ordinal      

        assert self.enc_type in [0, 2, 3], "model encoding type can only be 0, 2, 3"

        if self.enc_type in [2, 3]:
            self.rv_tran = RVTran(enc_type=enc_type, enc_f_type=enc_f_type)
            out_n = self.rv_tran.encoded_n
        else:
            self.rv_tran = None
            out_n = 1

        if not self.oned_input:            
            c, h, w = input_shape
            self.conv1 = nn.Conv2d(
                in_channels=c, out_channels=c // 2, kernel_size=3, padding="same"
            )
            self.conv2 = nn.Conv2d(
                in_channels=c // 2, out_channels=c // 4, kernel_size=3, padding="same"
            )
            fc_in = h * w * (c // 4)
        else:
            self.res = OneDResBlock(input_shape[0])
            fc_in = self.input_shape[0]

        if predict_v_pi:
            self.fc_logits = nn.Linear(fc_in, self.dim_actions*(self.num_actions if self.discrete_action else 2))
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
        
        if self.ordinal:
            indices = torch.arange(self.num_actions).view(-1, 1)
            ordinal_mask = (indices + indices.T) <= (self.num_actions - 1)
            ordinal_mask = ordinal_mask.float()
            self.register_buffer("ordinal_mask", ordinal_mask)

    def forward(self, h, predict_reward=True):
        x = h
        b = x.shape[0]
        if not self.oned_input:
            x_ = F.relu(self.conv1(x))
            x_ = F.relu(self.conv2(x_))
            x_ = torch.flatten(x_, start_dim=1)
        else:
            x_ = self.res(x)
        x_v, x_policy, x_done = x_, x_, x_

        if self.predict_v_pi:
            policy = self.fc_logits(x_policy)
            policy = policy.view(b, self.dim_actions, -1)
            if self.ordinal and self.discrete_action:
                norm_softm = F.sigmoid(policy)
                norm_softm_tiled = torch.tile(norm_softm.unsqueeze(-1), [1,1,1,self.num_actions])
                policy = torch.sum(torch.log(norm_softm_tiled + 1e-8) * self.ordinal_mask + torch.log(1 - norm_softm_tiled + 1e-8) * (1 - self.ordinal_mask), dim=-1)

            if self.enc_type in [2, 3]:
                v_enc_logit = self.fc_v(x_v)
                v_enc_v = F.softmax(v_enc_logit, dim=-1)
                v = self.rv_tran.decode(v_enc_v)
            else:
                v_enc_logit = None
                v = self.fc_v(x_v).squeeze(-1)
        else:
            v, v_enc_logit, policy = None, None, None

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
            policy=policy,
        )
        return out

class SRNet(nn.Module):
    def __init__(self, obs_shape, action_space, flags, frame_stack_n=1):
        super(SRNet, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.oned_input = len(obs_shape) == 1
        self.action_space = action_space
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)
        
        self.enc_type = flags.model_enc_type
        self.size_nn = flags.model_size_nn
        self.downscale_c = flags.model_downscale_c        
        self.noise_enable = flags.noise_enable
        self.has_memory = flags.model_has_memory

        self.decoder_depth = flags.model_decoder_depth
        if self.decoder_depth == 0:
            self.frame_stack_n = frame_stack_n        
        else:
            self.frame_stack_n = 1
        assert self.obs_shape[0] % self.frame_stack_n == 0, \
            f"obs channel {self.obs_shape[0]} should be divisible by frame stacking number {self.frame_stack_n}"        
        self.copy_n = self.obs_shape[0] // self.frame_stack_n

        self.encoder = FrameEncoder(
            prefix="sr",
            dim_rep_actions=self.dim_rep_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=True,
            decoder_depth=flags.model_decoder_depth,
            frame_stack_n=self.frame_stack_n,
            has_memory=self.has_memory,
        )
        self.per_state_len = len(self.encoder.initial_state())
        self.hidden_shape = self.encoder.out_shape
        inplanes = self.hidden_shape[0]
        if self.noise_enable:
            inplanes += self.flags.noise_n
        self.RNN = DynamicModel(
            dim_rep_actions=self.dim_rep_actions,
            inplanes=inplanes,
            oned_input=self.oned_input,
            size_nn=self.size_nn,
            outplanes=self.hidden_shape[0], 
            disable_half_grad=True,
            disable_bn=self.flags.model_disable_bn,
        )
        self.out = OutputNet(
            action_space=self.action_space,
            input_shape=self.hidden_shape,
            enc_type=self.enc_type,
            enc_f_type=self.flags.model_enc_f_type,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=False,
            predict_r=True,
            predict_done=self.flags.model_done_loss_cost > 0.0,
            ordinal=self.flags.model_ordinal,
        )
        self.rv_tran = self.out.rv_tran

        if self.noise_enable:
            x_shape = self.encoder.out_shape
            pre_in_shape = list(x_shape)
            pre_in_shape[0] = x_shape[0] + self.actions_ch
            self.noise_pre = NoiseModel(
                in_shape = pre_in_shape,
                size_nn = 1,
                noise_n = flags.noise_n,
                noise_d = flags.noise_d,
            )
            post_in_shape = list(x_shape)
            post_in_shape[0] = 2 * x_shape[0] + self.actions_ch
            self.noise_post = NoiseModel(
                in_shape = post_in_shape,
                size_nn = 1,
                noise_n = flags.noise_n,
                noise_d = flags.noise_d,
            )
            self.noise_n = flags.noise_n
            self.noise_d = flags.noise_d
            self.noise_alpha = flags.noise_alpha
            self.noise_mlp = flags.noise_mlp
            if self.noise_mlp:
                self.noise_mlp_net = MLP(
                    input_size=self.noise_n*self.noise_d,
                    layer_sizes=[flags.noise_n*2],
                    output_size=flags.noise_n,
                    output_activation=nn.ReLU,
                    norm=False
                )

    def initial_state(self, batch_size=1, device=None):
        return self.encoder.initial_state(batch_size=batch_size, device=device)

    def forward(self, env_state_norm, done, actions, state, one_hot=False, future_env_state_norm=None):
        """
        Args:
            env_state_norm (tensor): normalized env state (float) with shape (B, C, H, W), in the form of s_t
            done(tensor): done (bool) with shape (B,), in the form of d_t
            actions(tensor): action (int64) with shape (k+1, B, D, *), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            SRNetOut tuple with predicted rewards (rs), images (xs), done (dones) in the shape of (k, B, ...);
                in the form of y_{t+1}, y_{t+2}, ..., y_{t+k} and states with element in the shape of (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        actions = util.encode_action(actions, self.action_space, one_hot)       
        new_state = {}
        h, enc_state = self.encoder(env_state_norm, done, actions[0], state=state)
        new_state.update(enc_state)

        hs = [h.unsqueeze(0)]

        if self.noise_enable and future_env_state_norm is not None:
            assert future_env_state_norm.shape[:2] == (k, b)
            future_enc, _ = self.encoder(future_env_state_norm, None, actions[1:], enc_state, flatten=True)
            future_enc = future_enc.view((k, b) + future_enc.shape[1:])

        noise_losses = []

        for t in range(1, k + 1):
            if self.noise_enable:
                if future_env_state_norm is not None:
                    future_enc = future_enc[t-1]
                else:
                    future_enc = None
                h, noise_loss = self.compute_noise(h, actions[t], future_enc)
                noise_losses.append(noise_loss)
            h = self.RNN(h=h, actions=actions[t])
            hs.append(h.unsqueeze(0))
        hs = torch.concat(hs, dim=0)

        if future_env_state_norm is not None and self.noise_enable:
            noise_loss = torch.stack(noise_losses, dim=0)
        else:
            noise_loss = None

        new_state["sr_h"] = h
        if len(hs) > 1:
            xs = self.encoder.decode(hs[1:], flatten=True)
            if self.frame_stack_n > 1:
                stacked_x = env_state_norm
                stacked_xs = []
                for i in range(k):
                    stacked_x = torch.concat([stacked_x[:, self.copy_n:], xs[i]], dim=1)
                    stacked_xs.append(stacked_x)
                xs = torch.stack(stacked_xs, dim=0)
                new_state["last_x"] = stacked_x[:, self.copy_n:]   
       
        else:
            xs = None            
            if self.frame_stack_n > 1:
                new_state["last_x"] = env_state_norm[:, self.copy_n:].clone()

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
            state=new_state,
            noise_loss=noise_loss,
        )

    def forward_single(self, action, state, one_hot=False, future_x=None):
        """
        Single unroll of the network with one action
        Args:
            action(tensor): action (int64) with shape (B, *)
            one_hot (bool): whether to the action use one-hot encoding
        """
        new_state = {}
        new_state.update({k:v for k, v in state.items() if k.startswith(f"per_sr")})
        action = util.encode_action(action, self.action_space, one_hot)            
        h = state["sr_h"]        
        
        if self.noise_enable:
            if future_x is not None:
                future_enc_x, enc_state = self.encoder(future_x, None, action, state)
                new_state.update(enc_state)
            else:                
                future_enc_x = None
            h, _ = self.compute_noise(h, action, future_enc_x)
        h = self.RNN(h=h, actions=action)
        x = self.encoder.decode(h, flatten=False)
        if self.frame_stack_n > 1:
            x = torch.concat([state["last_x"], x], dim=1)

        out = self.out(h, predict_reward=True)
        new_state["sr_h"] = h
        if self.frame_stack_n > 1:
            new_state["last_x"] = x[:, self.copy_n:].clone()
        
        xs = util.safe_unsqueeze(x, 0)

        return SRNetOut(
            rs=util.safe_unsqueeze(out.rs, 0),
            r_enc_logits=util.safe_unsqueeze(out.r_enc_logits, 0),
            dones=util.safe_unsqueeze(out.dones, 0),
            done_logits=util.safe_unsqueeze(out.done_logits, 0),
            xs=xs,
            hs=util.safe_unsqueeze(h, 0),
            state=new_state,
            noise_loss=None,
        )
        
    def compute_noise(self, h, action, future_enc = None):   
        b = h.shape[0]   
        a = (
            action.unsqueeze(-1).unsqueeze(-1).tile([1, 1, h.shape[2], h.shape[3]])
        )
        noise_pre_in = torch.concat([h, a], dim=1)
        noise_pre_logit = self.noise_pre(noise_pre_in)
        
        if future_enc is not None:
            # training mode
            noise_post_in = torch.concat([noise_pre_in, future_enc], dim=1) 
            noise_post_logit = self.noise_post(noise_post_in)
            noise_logit = noise_post_logit
        else:
            # inference mode
            noise_post_logit = None
            noise_logit = noise_pre_logit

        noise_p = F.softmax(noise_logit, dim=-1)
        noise = torch.multinomial(noise_p.view(b*self.noise_n, self.noise_d), num_samples=1).view(b, self.noise_n)
        noise = F.one_hot(noise, num_classes=self.noise_d)
        noise = noise.detach() + noise_p - noise_p.detach()
        if not self.noise_mlp:
            noise = torch.sum(noise * torch.arange(self.noise_d, device=noise.device).float(), dim=-1) / self.noise_d
        else:
            noise = self.noise_mlp_net(noise.view(b, self.noise_n*self.noise_d))
        noise =  (
            noise.unsqueeze(-1).unsqueeze(-1).tile([1, 1, h.shape[2], h.shape[3]])
        )
        h = torch.concat([h, noise], dim=1)
        
        if future_enc is not None:
            log_pre = F.log_softmax(noise_pre_logit, dim=-1)
            log_post = F.log_softmax(noise_post_logit, dim=-1)
            noise_loss = self.noise_alpha * torch.sum(F.kl_div(log_post.detach(), log_pre, reduction='none', log_target=True), dim=(-1, -2))
            noise_loss += (1 - self.noise_alpha) * torch.sum(F.kl_div(log_post, log_pre.detach(), reduction='none', log_target=True), dim=(-1, -2))
        else:
            noise_loss = None
        
        return h, noise_loss    

class VPNet(nn.Module):
    def __init__(self, obs_shape, action_space, flags):
        super(VPNet, self).__init__()
        self.flags = flags
        self.obs_shape = obs_shape
        self.oned_input = len(self.obs_shape) == 1
        self.action_space = action_space
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)  
        self.enc_type = flags.model_enc_type
        self.has_memory = flags.model_has_memory
        self.size_nn = flags.model_size_nn # size_nn: int to adjust for size of model net
        self.downscale_c = flags.model_downscale_c_vp # downscale_c: int to downscale number of channels; default=2
        self.use_rnn = not util.check_perfect_model(flags.wrapper_type) # dont use rnn if we have perfect dynamic
        self.dual_net = flags.dual_net # rnn receives z only when we are using dual net
        self.predict_rd = (
           not flags.dual_net and self.use_rnn
        )  # network also predicts reward and done if not dual net under non-perfect dynamic   
        self.decoder_depth = flags.model_decoder_depth

        self.encoder = FrameEncoder(
            prefix="vr",
            dim_rep_actions=self.dim_rep_actions,
            input_shape=obs_shape,
            size_nn=self.size_nn,
            downscale_c=self.downscale_c,
            decoder=False,
            decoder_depth=0,
            has_memory=self.has_memory,
        )
        self.hidden_shape = self.encoder.out_shape
        inplanes = self.hidden_shape[0]
        
        if self.use_rnn:
            self.RNN = DynamicModel(
                dim_rep_actions=self.dim_rep_actions,
                inplanes=inplanes * 2 if self.dual_net else inplanes,
                oned_input=self.oned_input, 
                outplanes=inplanes,
                size_nn=self.size_nn,
                disable_half_grad=False,
                disable_bn=self.flags.model_disable_bn,
            )
        self.out = OutputNet(
            action_space=action_space,
            input_shape=self.hidden_shape,
            enc_type=self.enc_type,
            enc_f_type=self.flags.model_enc_f_type,
            zero_init=flags.model_zero_init,
            size_nn=self.size_nn,
            predict_v_pi=True,
            predict_r=self.predict_rd,
            predict_done=self.predict_rd and self.flags.model_done_loss_cost > 0.0,
            ordinal=self.flags.model_ordinal,
        )

        if not self.dual_net:
            if not self.oned_input:
                self.h_to_z_nn = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=False),
                    conv3x3(inplanes, inplanes),
                )
                self.z_to_h_nn = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=False),
                    conv3x3(inplanes, inplanes),
                )
            else:
                self.h_to_z_nn = OneDResBlock(hidden_size=inplanes)
                self.z_to_h_nn = OneDResBlock(hidden_size=inplanes)

        self.rv_tran = self.out.rv_tran

    def initial_state(self, batch_size=1, device=None):
        return self.encoder.initial_state(batch_size=batch_size, device=device)

    def h_to_z(self, h, flatten=False):
        if flatten:
            h_ = torch.flatten(h, 0, 1)
        else:
            h_ = h
        z = self.h_to_z_nn(h_)
        if flatten:
            z = z.view(h.shape[:2] + z.shape[1:])
        return z

    def z_to_h(self, z, flatten=False):
        if flatten:
            z_ = torch.flatten(z, 0, 1)
        else:
            z_ = z
        h = self.z_to_h_nn(z_)
        if flatten:
            h = h.view(z.shape[:2] + h.shape[1:])
        return h

    def forward(self, env_state_norm, x0, xs, done, actions, state, one_hot=False):
        """
        Args:
            env_state(tensor): normalized env tate with shape (B, C, H, W) in the form of s_t
            xs(tensor): output from SR-net with shape (k, B, C, H, W) in the form of x_{t+1}, ..., x_{t+k}; or x_{t}
            done(tensor): done (bool) with shape (B,), in the form of d_t
            actions(tensor): action (int64) with shape (k+1, B, D, *), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
        Return:
            VPNetOut tuple with predicted values (vs), policies (logits) in the shape of (k+1, B, ...);
                in the form of y_{t}, y_{t+1}, y_{t+2}, ..., y_{t+k} and states with element in the shape of (B, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        device = actions.device
        assert env_state_norm is not None or x0 is not None
        actions = util.encode_action(actions, self.action_space, one_hot)
        new_state = {}
        if done is None:
            done = torch.zeros(b, dtype=torch.bool, device=device)
        if k == 0:
            done = done.unsqueeze(0)
        else:
            done = torch.concat([done.unsqueeze(0), torch.zeros(k, b, dtype=torch.bool, device=xs.device)], dim=0)        
        
        if not self.dual_net:
            enc_in = env_state_norm.unsqueeze(0).detach()
            zs, enc_state = self.encoder(enc_in, done, actions[:1], state, flatten=True)
        else:
            if x0 is None:
                x0 = self.encoder.forward_pre_mem(env_state_norm, actions[0], end_depth=self.decoder_depth)
            full_xs = x0.unsqueeze(0)
            if k > 0: full_xs = torch.concat([full_xs, xs.detach()], dim=0)
            zs, enc_state = self.encoder(full_xs, done, actions, state, flatten=True, depth=self.decoder_depth)        
        new_state.update(enc_state)

        if self.use_rnn:
            if not self.dual_net:
                h = self.z_to_h(zs[0], flatten=False)
            else:
                h = torch.zeros(size=(b,) + self.hidden_shape, device=device)
                rnn_in = torch.concat([h, zs[0]], dim=1)
                h = self.RNN(h=rnn_in, actions=actions[0])
            hs = [h.unsqueeze(0)]
            for t in range(1, k + 1):
                if not self.dual_net:
                    rnn_in = h
                else:
                    rnn_in = torch.concat([h, zs[t]], dim=1)
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

        if not self.dual_net:
            pred_zs = torch.concat([zs[[0]], self.h_to_z(hs[1:], flatten=True)], dim=0)
        else:
            pred_zs = zs

        new_state["vp_h"] = h
        return VPNetOut(
            rs=util.safe_concat(outs[1:], "rs", 0),
            r_enc_logits=util.safe_concat(outs[1:], "r_enc_logits", 0),
            dones=util.safe_concat(outs[1:], "dones", 0),
            done_logits=util.safe_concat(outs[1:], "done_logits", 0),
            vs=util.safe_concat(outs, "vs", 0),
            v_enc_logits=util.safe_concat(outs, "v_enc_logits", 0),
            policy=util.safe_concat(outs, "policy", 0),
            hs=hs,
            true_zs=zs,
            pred_zs=pred_zs,
            state=new_state,
        )

    def forward_single(self, action, state, x=None, one_hot=False):
        """
        Single unroll of the network with one action
        Args:
            x(tensor): output from SR-net (float) with shape (B, *)
            action(tensor): action (int64) with shape (B, D, *)
            one_hot (bool): whether to the action use one-hot encoding
        """
        new_state = {}
        action = util.encode_action(action, self.action_space, one_hot)          
        if self.dual_net:
            z, enc_state = self.encoder(x, None, action, state, flatten=False, depth=self.decoder_depth)
            new_state.update(enc_state)
            rnn_in = torch.concat([state["vp_h"], z], dim=1)
        else:
            rnn_in = state["vp_h"]
        h = self.RNN(h=rnn_in, actions=action)
        out = self.out(h, predict_reward=True)
        new_state["vp_h"] = h

        if not self.dual_net:
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
            policy=util.safe_unsqueeze(out.policy, 0),
            hs=util.safe_unsqueeze(h, 0),
            true_zs=None,
            pred_zs=util.safe_unsqueeze(pred_z, 0),
            state=new_state,
        )
    
    def compute_z0(self, env_state_norm, done, action, state):
        b, *_ = action.shape
        device = env_state_norm.device
        if done is None:
            done = torch.zeros(b, dtype=torch.bool, device=device)        
        enc_in = env_state_norm.unsqueeze(0)
        zs, enc_state = self.encoder(enc_in.detach(), done, action.unsqueeze(0), state, flatten=True)
        return zs[0], enc_state

class ModelNet(BaseNet):
    def __init__(self, obs_space, action_space, flags, frame_stack_n=1):
        super(ModelNet, self).__init__()
        self.rnn = False
        self.flags = flags
        self.obs_shape = obs_space.shape
        self.action_space = action_space
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)  
        self.oned_input = len(self.obs_shape) == 1        
        self.enc_type = flags.model_enc_type
        self.size_nn = flags.model_size_nn
        self.dual_net = flags.dual_net
        self.reward_clip = flags.reward_clip
        self.value_clip = flags.reward_clip / (1 - flags.discounting)

        if obs_space.dtype == 'uint8':
            self.state_dtype_n = 0
        elif obs_space.dtype == 'float32':
            self.state_dtype_n = 1
        else:
            raise Exception(f"Unupported observation sapce", obs_space)
        
        low = torch.tensor(obs_space.low)
        high = torch.tensor(obs_space.high)
        self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()
        
        if self.need_norm:
            self.register_buffer("norm_low", low)
            self.register_buffer("norm_high", high)        

        self.vp_net = VPNet(self.obs_shape, action_space, flags)
        self.hidden_shape = list(self.vp_net.hidden_shape)
        if self.dual_net:
            self.sr_net = SRNet(self.obs_shape, action_space, flags, frame_stack_n)
            self.hidden_shape[0] += self.sr_net.hidden_shape[0]
        self.copy_n = self.obs_shape[0] // frame_stack_n
        self.decoder_depth = flags.model_decoder_depth

    def initial_state(self, batch_size=1, device=None):
        state = {}
        if self.dual_net:
            state.update(self.sr_net.initial_state(batch_size=batch_size, device=device))
        state.update(self.vp_net.initial_state(batch_size=batch_size, device=device))
        return state

    def normalize(self, x):
        if self.state_dtype_n == 0: assert x.dtype == torch.uint8
        if self.state_dtype_n == 1: assert x.dtype == torch.float32
        if self.need_norm:
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32        
        if self.need_norm:
            ch = x.shape[-3]
            x = torch.clamp(x, 0, 1)
            x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
            if self.state_dtype_n == 0: x = x.to(torch.uint8)
        return x

    def forward(self, env_state, done, actions, state, future_env_state=None):
        """
        Args:
            env_state(tensor): starting frame (uint if normalize else float) with shape (B, C, H, W)
            done(tensor): done (bool) with shape (B,), in the form of d_t
            actions(tensor): action (int64) with shape (k+1, B, D), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            normalize (tensor): whether to normalize x 
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            done(tensor): predicted done with shape (k, B, ...), in the form of d_{t+1}, d_{t+2}, ..., d_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            policy(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            state(dict): recurrent hidden state with shape (B, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        new_state = {}

        env_state_norm = self.normalize(env_state)        
        future_env_state_norm = self.normalize(future_env_state) if future_env_state is not None else None

        if self.dual_net:
            action = util.encode_action(actions[0], self.vp_net.action_space, one_hot=False)       
            x0 = self.vp_net.encoder.forward_pre_mem(env_state_norm, action, end_depth=self.decoder_depth)
            sr_net_out = self.sr_net(env_state_norm, done, actions, state, future_env_state_norm=future_env_state_norm)
            xs = sr_net_out.xs
            new_state.update(sr_net_out.state)
            full_xs = x0.unsqueeze(0)
            if k > 0: full_xs = torch.concat([full_xs, sr_net_out.xs], dim=0)
        else:
            sr_net_out = None
            x0 = None
            xs = None     
            full_xs = self.normalize(env_state).unsqueeze(0)

        vp_net_out = self.vp_net(env_state_norm, x0, xs, done, actions, state)
        new_state.update(vp_net_out.state)
        return self._prepare_out(sr_net_out, vp_net_out, new_state, full_xs)

    def forward_single(self, state, action, future_x=None):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            state(dict): recurrent state of the network
            action(tuple): action (int64) with shape (B)
        """
        state_ = {}
        if self.dual_net:
            sr_net_out = self.sr_net.forward_single(
                action=action, state=state, future_x=future_x
            )
            xs = sr_net_out.xs 
            x = xs[0]
            state_.update(sr_net_out.state)
        else:
            x = None
        vp_net_out = self.vp_net.forward_single(
            action=action, state=state, x=x,
        )
        state_.update(vp_net_out.state)
        return self._prepare_out(sr_net_out, vp_net_out, state_, xs)

    def _prepare_out(self, sr_net_out, vp_net_out,  state, xs):
        rd_out = sr_net_out if self.dual_net else vp_net_out
        if self.dual_net:
            hs = torch.concat([sr_net_out.hs, vp_net_out.hs], dim=2)
        else:
            hs = vp_net_out.hs
        rs = rd_out.rs
        vs = vp_net_out.vs
        if self.reward_clip > 0.:
            if rs is not None:
                rs = torch.clamp(rs, -self.reward_clip, +self.reward_clip)            
            assert not vs.requires_grad, "grad needs to be disabled at inference mode"
            vs = torch.clamp(vs, -self.value_clip, +self.value_clip)
        return DualNetOut(
            rs=rs,
            dones=rd_out.dones,
            vs=vs,
            v_enc_logits=vp_net_out.v_enc_logits,
            policy=vp_net_out.policy,
            xs=xs,
            hs=hs,
            zs=vp_net_out.pred_zs,
            state=state,
        )
    
    def compute_vs_loss(self, vs, v_enc_logits, target_vs):
        k, b = target_vs.shape
        if self.enc_type == 0:
            vs_loss = (vs[:k] - target_vs.detach()) ** 2
        else:
            target_vs_enc_v = self.vp_net.rv_tran.encode(target_vs)
            vs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(v_enc_logits[:k], 0, 1),
                target=torch.flatten(target_vs_enc_v.detach(), 0, 1),
            )
            vs_loss = vs_loss.view(k, b)
        return vs_loss

class RVTran(nn.Module):
    def __init__(self, enc_type, enc_f_type=0, eps=0.001):
        super(RVTran, self).__init__()
        assert enc_type in [
            1,
            2,
            3,
        ], f"only enc_type [1, 2, 3] is supported, not {enc_type}"
        self.enc_f_type = enc_f_type # 0 for MuZero encoding, 1 for Dreamer (symexp)
        self.support = 300 if self.enc_f_type == 0 else 20
        self.eps = eps
        self.enc_type = enc_type
        if self.enc_type == 2:
            atom_vector = self.decode_s(torch.arange(-self.support, self.support + 1, 1).float())
            self.register_buffer("atom_vector", atom_vector)
            self.encoded_n = 2 * self.support + 1
        elif self.enc_type == 3:
            atom_vector = torch.arange(-self.support, self.support + 1, 1)
            self.register_buffer("atom_vector", atom_vector)
            self.encoded_n = 2 * self.support + 1

    def forward(self, x):
        """encode the unencoded scalar reward or values to encoded scalar / vector according to MuZero"""
        with torch.no_grad():
            if self.enc_type == 1:
                enc = self.encode_s(x)
            elif self.enc_type == 2:
                enc = self.vector_enc(x)
            elif self.enc_type == 3:
                enc_s = self.encode_s(x)
                enc = self.vector_enc(enc_s)
            return enc
        
    def vector_enc(self, x):
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
        if self.enc_f_type == 0:
            return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + self.eps * x
        else:
            return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def decode_s(self, x):
        if self.enc_f_type == 0:
            return torch.sign(x) * (
                torch.square(
                    (torch.sqrt(1 + 4 * self.eps * (torch.abs(x) + 1 + self.eps)) - 1)
                    / (2 * self.eps)
                )
                - 1
            )
        else:
            return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
