from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from thinker.core.rnn import ConvAttnLSTM
from thinker.net import *


class LegacyActorNet(nn.Module):
    def __init__(self, obs_shape, gym_obs_shape, num_actions, flags):
        super(LegacyActorNet, self).__init__()
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
        self.conv_out = flags.tran_dim  # size of transformer / LSTM embedding dim
        self.num_rewards = 1 + int(flags.im_cost > 0.0) + int(flags.cur_cost > 0.0)
        # self.actor_see_p = flags.actor_see_p         # probability of allowing actor to see state
        self.actor_see_p = 1 if flags.actor_see_type >= 0 else 0
        # self.actor_see_encode = flags.actor_see_encode # Whether the actor see the model encoded state or the raw env state
        self.actor_see_encode = flags.actor_see_type >= 1
        self.actor_see_double_encode = (
            flags.actor_see_double_encode
        )  # Whether the actor see the model encoded state or the raw env state
        self.actor_drc = flags.actor_drc  # Whether to use drc in encoding state
        self.rnn_grad_scale = flags.rnn_grad_scale  # Grad scale for hidden state in RNN
        self.enc_type = flags.actor_enc_type
        self.model_type_nn = flags.model_type_nn
        self.model_size_nn = flags.model_size_nn

        self.conv_out_hw = 1
        self.d_model = self.conv_out

        self.conv1 = nn.Conv2d(
            in_channels=self.obs_shape[0],
            out_channels=self.conv_out // 2,
            kernel_size=1,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv_out // 2,
            out_channels=self.conv_out,
            kernel_size=1,
            stride=1,
        )
        self.frame_conv = torch.nn.Sequential(
            self.conv1, nn.ReLU(), self.conv2, nn.ReLU()
        )
        self.env_input_size = self.conv_out
        d_in = self.env_input_size + self.d_model

        self.core = ConvAttnLSTM(
            h=self.conv_out_hw,
            w=self.conv_out_hw,
            input_dim=d_in - self.d_model,
            hidden_dim=self.d_model,
            kernel_size=1,
            num_layers=self.tran_layer_n,
            num_heads=8,
            mem_n=self.tran_mem_n,
            attn=not self.tran_lstm_no_attn,
            attn_mask_b=self.attn_mask_b,
            grad_scale=self.rnn_grad_scale,
        )

        rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model
        self.fc = nn.Linear(rnn_out_size, 256)

        last_out_size = 256
        if self.actor_see_p > 0:
            down_scale_c = flags.model_downscale_c if self.actor_see_encode else 4
            last_out_size = int(
                last_out_size
                + (
                    256
                    if self.actor_drc
                    else (256 // down_scale_c // 4)
                    * (gym_obs_shape[1] // 16)
                    * (gym_obs_shape[2] // 16)
                )
            )
        self.im_policy = nn.Linear(last_out_size, self.num_actions)
        self.policy = nn.Linear(last_out_size, self.num_actions)

        if self.enc_type in [1, 2]:
            self.rv_tran = RVTran(vec=self.enc_type == 2)
        else:
            self.rv_tran = None
        if self.enc_type in [0, 1]:
            self.baseline = nn.Linear(last_out_size, self.num_rewards)
        else:
            self.out_n = self.rv_tran.encoded_n
            self.baseline = nn.Linear(last_out_size, self.num_rewards * self.out_n)

        self.reset = nn.Linear(last_out_size, 2)

        if self.actor_see_p > 0:
            if not self.actor_drc:
                if not self.actor_see_encode:
                    self.gym_frame_encoder = FrameEncoder(
                        frame_channels=gym_obs_shape[0],
                        num_actions=self.num_actions,
                        down_scale_c=down_scale_c,
                        concat_action=False,
                    )
                if self.model_type_nn in [2, 3] and self.actor_see_encode:
                    in_channels = 64 if self.model_type_nn == 2 else 128
                else:
                    in_channels = int(256 // down_scale_c)
                if self.actor_see_double_encode:
                    in_channels = in_channels * 2
                self.gym_frame_conv = torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=int(256 // down_scale_c // 2),
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=int(256 // down_scale_c // 2),
                        out_channels=int(256 // down_scale_c // 4),
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                )
            else:
                assert (
                    not self.actor_see_encode
                ), "actor_drc is not compatiable with actor_see_encode"
                self.gym_frame_conv = torch.nn.Sequential(
                    nn.Conv2d(
                        in_channels=gym_obs_shape[0],
                        out_channels=32,
                        kernel_size=8,
                        stride=4,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                )
                compute_hw_out = (
                    lambda hw_in, kernel_size, stride: (hw_in - (kernel_size - 1) - 1)
                    // stride
                    + 1
                )
                hw_out_1 = compute_hw_out(gym_obs_shape[1], 8, 4)
                hw_out_2 = compute_hw_out(hw_out_1, 4, 2)
                self.conv_out_hw_2 = hw_out_2
                self.drc_core = ConvAttnLSTM(
                    h=hw_out_2,
                    w=hw_out_2,
                    input_dim=32,
                    hidden_dim=32,
                    kernel_size=3,
                    num_layers=3,
                    num_heads=8,
                    mem_n=0,
                    attn=False,
                    attn_mask_b=0.0,
                    grad_scale=self.rnn_grad_scale,
                )
                self.drc_fc = nn.Linear(hw_out_2 * hw_out_2 * 32, 256)

        # print("actor size: ", sum(p.numel() for p in self.parameters()))
        # for k, v in self.named_parameters(): print(k, v.numel())

        self.initial_state(1)  # just for setting core_state_sep_ind

    def initial_state(self, batch_size, device=None):
        state = self.core.init_state(batch_size, device=device)
        self.core_state_sep_ind = len(state)
        if self.actor_see_p > 0 and self.actor_drc:
            state = state + self.drc_core.init_state(batch_size, device=device)
        return state

    def forward(self, obs, core_state=()):
        """one-step forward for the actor;
        args:
            obs (EnvOut):
                model_out (tensor): model output with shape (T x B x C) or (B x C)
                done  (tensor): if episode ends with shape (T x B) or (B)
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

        x = obs.model_out.unsqueeze(-1).unsqueeze(-1)
        done = obs.done

        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        if len(done.shape) == 1:
            done = done.unsqueeze(0)

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        env_input = self.frame_conv(x)
        core_input = env_input.view(T, B, -1, self.conv_out_hw, self.conv_out_hw)
        core_output_list = []
        core_state_1 = core_state[: self.core_state_sep_ind]
        notdone = ~(done.bool())
        for n, (input, nd) in enumerate(zip(core_input.unbind(), notdone.unbind())):
            # Input shape: B, self.conv_out + self.num_actions + 1, H, W
            for t in range(self.tran_t):
                if t > 0:
                    nd = torch.ones(B).to(x.device).bool()
                nd = nd.view(-1)
                output, core_state_1 = self.core(
                    input, core_state_1, nd, nd
                )  # output shape: 1, B, core_output_size

            last_input = input
            core_output_list.append(output)
        core_output = torch.cat(core_output_list)
        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.flatten(core_output, start_dim=1)
        core_output = F.relu(self.fc(core_output))

        if self.actor_see_p > 0:
            if not self.actor_see_encode:
                gym_x = obs.gym_env_out.float()
                gym_x = gym_x * obs.see_mask.float().unsqueeze(-1).unsqueeze(
                    -1
                ).unsqueeze(-1)
                gym_x = torch.flatten(gym_x, 0, 1)
            if not self.actor_drc:
                if not self.actor_see_encode:
                    conv_out = self.gym_frame_encoder(gym_x, actions=None)
                else:
                    conv_out = torch.flatten(obs.model_encodes, 0, 1)
                conv_out = self.gym_frame_conv(conv_out)
                conv_out = torch.flatten(conv_out, start_dim=1)
                core_output = torch.concat([core_output, conv_out], dim=1)
            else:
                gym_x = gym_x / 255.0
                conv_out = self.gym_frame_conv(gym_x)
                core_input = conv_out.view(
                    T, B, -1, self.conv_out_hw_2, self.conv_out_hw_2
                )
                core_output_list = []
                core_state_2 = core_state[self.core_state_sep_ind :]
                for n, (input, nd) in enumerate(
                    zip(core_input.unbind(), notdone.unbind())
                ):
                    for t in range(3):
                        if t > 0:
                            nd = torch.ones(B).to(x.device).bool()
                        nd = nd.view(-1)
                        output, core_state_2 = self.drc_core(
                            input, core_state_2, nd, nd
                        )  # output shape: 1, B, core_output_size
                    core_output_list.append(output)

                core_output_2 = torch.cat(core_output_list)
                core_output_2 = torch.flatten(core_output_2, 0, 1)
                core_output_2 = torch.flatten(core_output_2, start_dim=1)
                core_output_2 = F.relu(self.drc_fc(core_output_2))
                core_output = torch.concat([core_output, core_output_2], dim=1)

        policy_logits = self.policy(core_output)
        im_policy_logits = self.im_policy(core_output)
        reset_policy_logits = self.reset(core_output)

        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        im_action = torch.multinomial(F.softmax(im_policy_logits, dim=1), num_samples=1)
        reset_action = torch.multinomial(
            F.softmax(reset_policy_logits, dim=1), num_samples=1
        )

        if self.enc_type == 0:
            baseline = self.baseline(core_output)
            baseline_enc = None
        elif self.enc_type == 1:
            baseline_enc_s = self.baseline(core_output)
            baseline = self.rv_tran.decode(baseline_enc_s)
            baseline_enc = baseline_enc_s
        else:
            baseline_enc_logit = self.baseline(core_output).reshape(
                T * B, self.num_rewards, self.out_n
            )
            baseline_enc_v = F.softmax(baseline_enc_logit, dim=-1)
            baseline = self.rv_tran.decode(baseline_enc_v)
            baseline_enc = baseline_enc_logit

        reg_loss = (
            1e-3 * torch.sum(policy_logits**2, dim=-1) / 2
            + 1e-5 * torch.sum(core_output**2, dim=-1) / 2
        )
        reg_loss = reg_loss.view(T, B)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        im_policy_logits = im_policy_logits.view(T, B, self.num_actions)
        reset_policy_logits = reset_policy_logits.view(T, B, 2)

        action = action.view(T, B)
        im_action = im_action.view(T, B)
        reset_action = reset_action.view(T, B)

        baseline_enc = (
            baseline_enc.view((T, B) + baseline_enc.shape[1:])
            if baseline_enc is not None
            else None
        )
        baseline = baseline.view(T, B, self.num_rewards)

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

        if self.actor_see_p > 0 and self.actor_drc:
            core_state = core_state_1 + core_state_2
        else:
            core_state = core_state_1
        return actor_out, core_state

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


class ModelNetRNN(nn.Module):
    def __init__(self, obs_shape, num_actions, flags):
        super(ModelNetRNN, self).__init__()
        self.rnn = True
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.tran_t = 1
        self.tran_mem_n = 0
        self.tran_layer_n = 1
        self.tran_lstm_no_attn = True
        self.attn_mask_b = 0

        self.conv_out = 32
        self.conv_out_hw = 8
        # self.conv1 = nn.Conv2d(in_channels=self.obs_shape[0], out_channels=self.conv_out, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(in_channels=self.conv_out, out_channels=self.conv_out, kernel_size=4, stride=2)
        # self.frame_conv = torch.nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())

        self.conv_out = 32
        self.conv_out_hw = 5
        self.frameEncoder = FrameEncoder(num_actions=self.num_actions)
        self.frame_conv = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128 // 2, kernel_size=3, padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128 // 2,
                out_channels=128 // 4,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
        )

        self.debug = flags.model_rnn_debug
        self.disable_mem = flags.model_disable_mem

        if self.debug:
            self.policy = nn.Linear(5 * 5 * 32, self.num_actions)
            self.baseline = nn.Linear(5 * 5 * 32, 1)
            self.r = nn.Linear(5 * 5 * 32, 1)
        else:
            self.env_input_size = self.conv_out
            self.d_model = self.conv_out
            d_in = self.env_input_size + self.d_model

            self.core = ConvAttnLSTM(
                h=self.conv_out_hw,
                w=self.conv_out_hw,
                input_dim=d_in - self.d_model,
                hidden_dim=self.d_model,
                kernel_size=3,
                num_layers=self.tran_layer_n,
                num_heads=8,
                mem_n=self.tran_mem_n,
                attn=not self.tran_lstm_no_attn,
                attn_mask_b=self.attn_mask_b,
                grad_scale=1,
            )

            rnn_out_size = self.conv_out_hw * self.conv_out_hw * self.d_model
            self.fc = nn.Linear(rnn_out_size, 256)

            self.policy = nn.Linear(256, self.num_actions)
            self.baseline = nn.Linear(256, 1)

    def init_state(self, bsz, device=None):
        if self.debug:
            return (torch.zeros(1, bsz, 1, 1, 1),)
        return self.core.init_state(bsz, device)

    def forward(self, x, actions, done, state, one_hot=False):
        """
        Args:
            x(tensor): frames (uint8 or float) with shape (T, B, C, H, W), in the form of s_t
            actions(tensor): action (int64) with shape (T, B) or (T, B, num_actions)
            done(tensor): done (bool) with shape (T, B)
            state(tuple): tuple of inital state
            one_hot(bool): whether the actions are in one-hot encoding
        Return:
            vs(tensor): values (float) with shape (T, B)
            logits(tensor): policy logits (float) with shape (T, B, num_actions)
            state(tuple): tuple of state tensor after the last step
        """
        assert done.dtype == torch.bool, "done has to be boolean"

        T, B = x.shape[0], x.shape[1]

        if one_hot:
            assert actions.shape == (T, B, self.num_actions), (
                "invalid action shape:",
                actions.shape,
            )
        else:
            assert actions.shape == (
                T,
                B,
            ), ("invalid action shape:", actions.shape)
        assert len(x.shape) == 5

        # x = x.float() / 255.0
        x = torch.flatten(x, 0, 1)
        if not one_hot:
            actions = F.one_hot(actions.view(T * B), self.num_actions).float()
        else:
            actions = actions.view(T * B, -1)

        # conv_out = self.frame_conv(x)
        conv_out = self.frameEncoder(x, actions)
        conv_out = self.frame_conv(conv_out)

        if self.debug:
            core_output = torch.flatten(conv_out, start_dim=1)
            vs = self.baseline(core_output).view(T, B)
            logits = self.policy(core_output).view(T, B, self.num_actions)
            state = self.init_state(B, x.device)
            return vs, logits, state

        core_input = conv_out
        core_input = core_input.view(
            T, B, self.env_input_size, self.conv_out_hw, self.conv_out_hw
        )
        core_output_list = []

        if self.disable_mem:
            state = self.init_state(bsz=B, device=x.device)
        notdone = (~done).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            for t in range(self.tran_t):
                nd_ = nd if t == 0 else torch.ones_like(nd)
                output, state = self.core(
                    input, state, nd_, nd_
                )  # output shape: 1, B, core_output_size
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        core_output = F.relu(self.fc(torch.flatten(core_output, start_dim=1)))
        vs = self.baseline(core_output).view(T, B)
        logits = self.policy(core_output).view(T, B, self.num_actions)
        return vs, logits, state

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


class ModelNetBase(nn.Module):
    def __init__(self, obs_shape, num_actions, flags):
        super(ModelNetBase, self).__init__()
        self.rnn = False
        self.flags = flags
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.reward_transform = flags.reward_transform
        self.hz_tran = flags.model_hz_tran
        self.kl_alpha = flags.model_kl_alpha
        self.type_nn = (
            flags.model_type_nn
        )  # type_nn: type of neural network for the model; 0 for small, 1 for large, 2 for small enet, 3 for large enet
        self.size_nn = (
            flags.model_size_nn
        )  # size_nn: int to adjust for size of model net (for model_type_nn == 3 only)
        self.frameEncoder = FrameEncoder(
            num_actions=num_actions,
            input_shape=obs_shape,
            type_nn=self.type_nn,
            size_nn=self.size_nn,
            decoder=self.flags.model_img_loss_cost > 0.0,
        )
        f_shape = self.frameEncoder.out_shape
        inplanes = f_shape[0] if self.type_nn not in [4] else f_shape[0] * f_shape[1]

        self.dynamicModel = DynamicModel(
            num_actions=num_actions,
            inplanes=inplanes,
            type_nn=self.type_nn,
            size_nn=self.size_nn,
            disable_bn=self.flags.model_disable_bn,
        )
        d_outplanes = self.dynamicModel.outplanes

        input_shape = (
            f_shape
            if self.type_nn not in [4]
            else (f_shape[0] * f_shape[1] + d_outplanes,)
        )

        self.output_rvpi = Output_rvpi(
            num_actions=num_actions,
            input_shape=input_shape,
            value_prefix=flags.value_prefix,
            max_unroll_length=flags.model_k_step_return,
            reward_transform=self.reward_transform,
            stop_vpi_grad=flags.model_stop_vpi_grad,
            zero_init=flags.model_zero_init,
            type_nn=self.type_nn,
            size_nn=self.size_nn,
            predict_done=self.flags.model_done_loss_cost > 0.0,
            disable_bn=self.flags.model_disable_bn,
        )

        if self.reward_transform:
            self.reward_tran = self.output_rvpi.reward_tran

        self.supervise = flags.model_sup_loss_cost > 0.0
        self.model_supervise_type = flags.model_supervise_type

        if self.supervise:
            if self.model_supervise_type == 0:
                flatten_in_dim = (obs_shape[1] // 16) * (obs_shape[2]) // 16 * inplanes
                self.P_1 = nn.Sequential(
                    nn.Linear(flatten_in_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                )
                self.P_2 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                )
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        if self.hz_tran:
            if self.type_nn in [0, 1, 2, 3]:
                self.h_to_z_conv = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=self.flags.model_disable_bn),
                    conv3x3(inplanes, inplanes),
                )
                self.z_to_h_conv = nn.Sequential(
                    ResBlock(inplanes=inplanes, disable_bn=self.flags.model_disable_bn),
                    conv3x3(inplanes, inplanes),
                )
            elif self.type_nn in [4]:
                self.h_to_z_mlp = mlp(
                    d_outplanes, [d_outplanes], inplanes, activation=nn.ELU, norm=False
                )

        assert not (
            self.type_nn == 4 and not self.hz_tran
        ), "dreamer net must be used with hz separation"

    def h_to_z(self, h, action=None, flatten=False):
        if not self.hz_tran:
            return h, None
        if flatten:
            h_ = torch.flatten(h, 0, 1)
        else:
            h_ = h
        if self.type_nn in [4]:
            b = h_.shape[0]
            s_size = 32 * self.size_nn
            z_logit = self.h_to_z_mlp(h).view(b, s_size, s_size)
            z_p = F.softmax(z_logit, dim=2)
            z = torch.multinomial(z_p.view(b * s_size, s_size), num_samples=1).view(
                b, s_size
            )
            z = F.one_hot(z, num_classes=s_size)
            z = z + z_p - z_p.detach()
        else:
            z = self.h_to_z_conv(h_)
            z_logit = None
        if flatten:
            z = z.view(h.shape[:2] + z.shape[1:])
            z_logit = (
                z_logit.view(h.shape[:2] + z_logit.shape[1:])
                if z_logit is not None
                else None
            )
        return z, z_logit

    def z_to_h(self, z, flatten=False):
        if self.type_nn in [4]:
            in_shape = z.shape[0] if not flatten else (z.shape[0] * z.shape[1])
            h = self.dynamicModel.init_state(in_shape, device=z.device)[0]
        else:
            if not self.hz_tran:
                return z
            if flatten:
                z_ = torch.flatten(z, 0, 1)
            else:
                z_ = z
            h = self.z_to_h_conv(z_)
        if flatten:
            h = h.view(z.shape[:2] + h.shape[1:])
        return h

    def supervise_loss(self, xs, model_net_out, is_weights, mask, one_hot=False):
        """
        Args:
            xs(tensor): state s with shape (k+1, B, *) in the form of s_t, s_{t+1}, ..., s_{t+k}
            model_net_out(tensor): model_net_out from forward
            mask(tensor): mask (float) with shape (k, B)
            im_weights(tensor): importance weight with shape (B) for each sample;
        Return:
            loss(tensor): scalar self-supervised loss
        """
        k, b, *_ = xs.shape
        k = k - 1

        if self.model_supervise_type in [0, 1, 2]:
            # 0 for SimSiam loss (efficient zero)
            # 1 for direct cos loss
            # 2 for direct L2 loss
            # 3 for dreamer loss
            true_zs = model_net_out.true_zs[1:]
            pred_zs = torch.flatten(model_net_out.pred_zs[1:], 0, 1)
            pred_zs = torch.flatten(pred_zs, 1)
            if self.model_supervise_type == 0:
                src = self.P_2(self.P_1(pred_zs))
            elif self.model_supervise_type in [1, 2]:
                src = pred_zs

            with torch.no_grad():
                true_zs = torch.flatten(true_zs, 0, 1)
                true_zs = torch.flatten(true_zs, 1)
                if self.model_supervise_type == 0:
                    tgt = self.P_1(true_zs)
                elif self.model_supervise_type in [1, 2]:
                    tgt = true_zs

            if self.model_supervise_type in [0, 1]:
                sup_loss = -self.cos(src, tgt.detach())
            elif self.model_supervise_type == 2:
                sup_loss = torch.mean((src - tgt.detach()) ** 2, dim=-1)
            sup_loss = sup_loss.view(k, b)
            s_mask = mask[1:]

        elif self.model_supervise_type in [3]:
            if self.flags.model_sup_ignore_first:
                true_z_logits = model_net_out.true_z_logits[1:]
                pred_z_logits = model_net_out.pred_z_logits[1:]
                k_ = k
            else:
                true_z_logits = model_net_out.true_z_logits
                pred_z_logits = model_net_out.pred_z_logits
                k_ = k + 1

            _, _, c1, c2 = true_z_logits.shape
            true_z_logits = true_z_logits.reshape(k_ * b * c1, c2)
            pred_z_logits = pred_z_logits.reshape(k_ * b * c1, c2)

            alpha = self.kl_alpha
            target = F.softmax(true_z_logits, dim=-1)
            sup_loss_pre = torch.nn.CrossEntropyLoss(reduction="none")(
                input=pred_z_logits, target=target.detach()
            )
            sup_loss_post = torch.nn.CrossEntropyLoss(reduction="none")(
                input=pred_z_logits.detach(), target=target
            )
            sup_loss = alpha * sup_loss_pre + (1 - alpha) * sup_loss_post
            sup_loss = sup_loss.view(k_, b, c1)
            sup_loss = torch.sum(sup_loss, dim=-1)

            if self.flags.model_sup_ignore_first:
                s_mask = mask[1:]
            else:
                s_mask = mask

        if mask is not None:
            sup_loss = sup_loss * s_mask
        sup_loss = torch.sum(sup_loss, dim=0)
        sup_loss = sup_loss * is_weights
        sup_loss = torch.sum(sup_loss)

        return sup_loss

    def img_loss(self, xs, model_net_out, is_weights, mask):
        """
        Args:
            xs(tensor): state s with shape (k+1, B, *) in the form of s_t, s_{t+1}, ..., s_{t+k}
            model_net_out(tensor): model_net_out from forward
            mask(tensor): mask (float) with shape (k, B)
            im_weights(tensor): importance weight with shape (B) for each sample;
        Return:
            loss(tensor): scalar img reconstruction loss
        """
        k, b, *_ = xs.shape
        k = k - 1

        if self.flags.model_img_loss_cost > 0:
            # image reconstruction loss

            if self.flags.model_sup_ignore_first:
                true_zs = model_net_out.true_zs[1:]
                pred_zs = model_net_out.pred_zs[1:]
                hs = model_net_out.hs[1:]
                xs = xs[1:]
            else:
                true_zs = model_net_out.true_zs
                pred_zs = model_net_out.pred_zs
                hs = model_net_out.hs
                xs = xs

            if self.flags.model_img_loss_use_pred_zs:
                decoder_in = pred_zs
            else:
                decoder_in = true_zs

            if self.type_nn in [4]:
                h = torch.flatten(hs, 0, 1)
            else:
                h = None

            pred_xs = self.frameEncoder.decode(torch.flatten(decoder_in, 0, 1), h)
            xs = torch.flatten(xs, 0, 1).float() / 255.0
            img_loss = torch.sum(torch.square(xs - pred_xs), dim=(1, 2, 3))
            img_loss = img_loss.view(
                k if self.flags.model_sup_ignore_first else k + 1, b
            )

            if mask is not None:
                if self.flags.model_sup_ignore_first:
                    i_mask = mask[1:]
                else:
                    i_mask = mask
                img_loss = img_loss * i_mask

            img_loss = torch.sum(img_loss, dim=0)
            img_loss = img_loss * is_weights
            img_loss = torch.sum(img_loss)
        else:
            img_loss = None

        return img_loss

    def forward(self, xs, actions, one_hot=False, compute_true_z=False, inference=True):
        """
        Args:
            x(tensor): frames (uint8) with shape (k+1, B, C, H, W), in the form of s_t, s_{t+1}, ..., s_{t+k}, or
                with shape (B, C, H, W) in the form of s_t;
            actions(tensor): action (int64) with shape (k+1, B), in the form of a_{t-1}, a_{t}, a_{t+1}, .. a_{t+k-1}
            one_hot (bool): whether to the action use one-hot encoding
            true_z(bool): if True, true z will be generated (require x to be in shape (k+1, ...))
            inference(bool): if True, predicted z will be fed back into the network, else true z
                 (require x to be in shape (k+1, ...))
        Return:
            rs(tensor): predicted reward with shape (k, B, ...), in the form of r_{t+1}, r_{t+2}, ..., r_{t+k}
            vs(tensor): predicted value with shape (k+1, B, ...), in the form of v_{t}, v_{t+1}, v_{t+2}, ..., v_{t+k}
            logits(tensor): predicted policy with shape (k+1, B, ...), in the form of pi_{t}, pi_{t+1}, pi_{t+2}, ..., pi_{t+k}
            pred_zs(tensor): predicted encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            true_zs(tensor): true encoded states with shape (k+1, B, ...), in the form of z_t, z_{t+1}, z_{t+2}, ..., z_{t+k}
            hs(tensor): hidden state with shape (k+1, B, ...), in the form of h_t, h_{t+1}, h_{t+2}, ..., h_{t+k}
            r_state(tensor): reward hidden state with shape (B, ...)
            (Recall we use the transition notation: s_t, a_t, r_{t+1}, s_{t+1}, ...)
        """
        k, b, *_ = actions.shape
        k = k - 1
        if len(xs.shape) == 4:
            xs = xs.unsqueeze(0)
        if compute_true_z or not inference:
            assert xs.shape[0] == k + 1, (
                "in non-inference or true_z mode, xs shape should be k+1 instead of %d"
                % xs.shape[0]
            )

        # initialise empty list
        data = {key: [] for key in ModelNetOut._fields}

        if not one_hot:
            actions = F.one_hot(actions, self.num_actions)
        if self.type_nn in [0, 1, 2, 3]:
            true_z, true_z_logit = self.frameEncoder(xs[0], actions[0], None)
            h = self.z_to_h(true_z, flatten=False)
        elif self.type_nn in [4]:
            h = self.dynamicModel.init_state(bsz=b, device=xs.device)[0]
            true_z, true_z_logit = self.frameEncoder(xs[0], actions[0], h)
        r_state = self.output_rvpi.init_state(bsz=xs.shape[1], device=xs.device)
        out_net_out = self.output_rvpi(true_z, h, predict_reward=False, state=())
        pred_z, pred_z_logit = self.h_to_z(h)

        data["vs"].append(out_net_out.vs)
        data["v_enc_logits"].append(out_net_out.v_enc_logits)
        data["logits"].append(out_net_out.logits)
        data["pred_zs"].append(pred_z)
        data["pred_z_logits"].append(pred_z_logit)
        data["true_zs"].append(true_z)
        data["true_z_logits"].append(true_z_logit)
        data["hs"].append(h)

        z_in = true_z

        for t in range(1, actions.shape[0]):
            out = self.forward_zh(z_in, h, r_state, actions[t])
            for key in out._fields:
                if key in ["true_zs", "true_z_logits", "r_state"]:
                    continue
                val = getattr(out, key)
                if val is not None:
                    val = val.squeeze(0)
                data[key].append(val)
            if compute_true_z or not inference:
                true_z, true_z_logit = self.frameEncoder(
                    xs[t], actions[t], data["hs"][t]
                )
                data["true_zs"].append(true_z)
                data["true_z_logits"].append(true_z_logit)
            z_in = out.pred_zs[-1] if inference else true_z
            h = out.hs[-1]
            r_state = out.r_state

        for key in data.keys():
            if len(data[key]) > 0 and data[key][0] is not None:
                data[key] = torch.concat([val.unsqueeze(0) for val in data[key]], dim=0)
            else:
                data[key] = None

        data["r_state"] = r_state
        return util.construct_tuple(ModelNetOut, **data)

    def forward_zh(self, z, h, r_state, action, one_hot=True):
        """
        One-step transition from z_t, h_t, a_t to predicted z_{t+1}, h_{t+1}, r_{t+1}, v_{t+1}, pi_{t+1}
        Args:
            z: encoded state with shape (B, ...), in the form of z_t
            h: hidden state with shape (B, ...), in the form of h_t
            r_state: hidden state of reward predictor with shape (B, ...)
            action: action with shape (B, ...)
        """
        if not one_hot:
            action = F.one_hot(action, self.num_actions)
        h = self.dynamicModel(z=z, h=h, actions=action)
        pred_z, pred_z_logit = self.h_to_z(h, action=action)
        out_net_out = self.output_rvpi(z, h, predict_reward=True, state=r_state)

        return ModelNetOut(
            single_rs=util.safe_unsqueeze(out_net_out.single_rs, 0),
            rs=util.safe_unsqueeze(out_net_out.rs, 0),
            r_enc_logits=util.safe_unsqueeze(out_net_out.r_enc_logits, 0),
            dones=util.safe_unsqueeze(out_net_out.dones, 0),
            done_logits=util.safe_unsqueeze(out_net_out.done_logits, 0),
            vs=util.safe_unsqueeze(out_net_out.vs, 0),
            v_enc_logits=util.safe_unsqueeze(out_net_out.v_enc_logits, 0),
            logits=util.safe_unsqueeze(out_net_out.logits, 0),
            pred_xs=None,
            pred_zs=util.safe_unsqueeze(pred_z, 0),
            pred_z_logits=util.safe_unsqueeze(pred_z_logit, 0),
            true_zs=None,
            true_z_logits=None,
            hs=util.safe_unsqueeze(h, 0),
            r_state=r_state,
        )

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            weights = {k: v.to(device) for k, v in weights.items()}
        self.load_state_dict(weights)
