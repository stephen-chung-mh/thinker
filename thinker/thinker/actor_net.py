from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from thinker import util
from thinker.core.rnn import ConvAttnLSTM
from thinker.core.module import MLP, OneDResBlock, tile_and_concat_tensors
from thinker.model_net import RVTran
from thinker.legacy import AFrameEncoderLegacy
from gym import spaces

ActorOut = namedtuple(
    "ActorOut",
    [     
        "pri", # sampled primiary action
        "pri_param", # parameter for primary action dist, can be logit or gaussian mean + log var        
        "reset", # sampled reset action
        "reset_logits", # parameter for reset dist, i.e. logit
        "action", # tuple of the above two actions 
        "action_prob", # prob of primary action 
        "c_action_log_prob", # log prob of chosen action
        "baseline", # baseline 
        "baseline_enc", # baseline encoding, only for non-scalar enc_type
        "entropy_loss", # entropy loss
        "reg_loss", # regularization loss
        "misc",
    ],
)

def compute_discrete_log_prob(logits, actions):
    assert len(logits.shape) == len(actions.shape) + 1
    has_dim = len(actions.shape) == 3    
    end_dim = 2 if has_dim else 1
    log_prob = -torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(logits, 0, end_dim), target=torch.flatten(actions, 0, end_dim)
    )
    log_prob = log_prob.view_as(actions)
    if has_dim:
        log_prob = torch.sum(log_prob, dim=-1)
    return log_prob


def sample(logits, greedy, dim=-1):
    if not greedy:
        gumbel_noise = torch.empty_like(logits).uniform_().clamp(1e-10, 1).log().neg_().clamp(1e-10, 1).log().neg_()
        sampled_action = (logits + gumbel_noise).argmax(dim=dim)
        return sampled_action.detach()
    else:
        return torch.argmax(logits, dim=dim)

def atanh(x, eps=1e-6):
    x = torch.clamp(x, -1.0+eps, 1.0-eps)
    return 0.5 * (x.log1p() - (-x).log1p())

class AFrameEncoder(nn.Module):
    # processor for 3d inputs; can be applied to model's hidden state or predicted real state
    def __init__(self, 
                 input_shape, 
                 flags,
                 downpool=False, 
                 firstpool=False,    
                 out_size=256,
                 see_double=False,                 
                 ):
        super(AFrameEncoder, self).__init__()
        if see_double:
            input_shape = (input_shape[0] // 2,) + tuple(input_shape[1:])
        self.input_shape = input_shape        
        self.downpool = downpool
        self.firstpool = firstpool
        self.out_size = out_size
        self.see_double = see_double    
        self.enc_1d_shallow = getattr(flags, "enc_1d_shallow", False)
        self.flags = flags    

        self.oned_input = len(self.input_shape) == 1
        if self.enc_1d_shallow and self.oned_input: self.out_size = 64

        in_channels = input_shape[0]
        if not self.oned_input:
            # following code is from Torchbeast, which is the same as Impala deep model            
            conv_out_h = input_shape[1]
            conv_out_w = input_shape[2]

            self.feat_convs = []
            self.resnet1 = []
            self.resnet2 = []
            self.convs = []

            if firstpool:
                self.down_pool_conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                in_channels = 16
                conv_out_h = (conv_out_h - 1) // 2 + 1
                conv_out_w = (conv_out_w - 1) // 2 + 1

            num_chs = [16, 32, 32] if downpool else [64, 64, 32]
            for num_ch in num_chs:
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
                if downpool:
                    feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                    conv_out_h = (conv_out_h - 1) // 2 + 1
                    conv_out_w = (conv_out_w - 1) // 2 + 1
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
            core_out_size = num_ch * conv_out_h * conv_out_w
        else:
            if not self.enc_1d_shallow:
                n_block = self.flags.enc_1d_block
                hidden_size = self.flags.enc_1d_hs
                self.hidden_size = hidden_size
                self.input_block = nn.Sequential(
                    nn.Linear(in_channels, hidden_size),
                    nn.ReLU()
                )            
                self.res = nn.Sequential(*[OneDResBlock(hidden_size, norm=self.flags.enc_1d_norm) for _ in range(n_block)])
                core_out_size = hidden_size
            else:
                self.input_block = nn.Sequential(nn.Linear(in_channels, 64), nn.Tanh())            
                self.res = nn.Identity()   
                core_out_size = 64      
        
        mlp_out_size = self.out_size if not self.see_double else self.out_size // 2
        self.fc = nn.Sequential(nn.Linear(core_out_size, mlp_out_size), nn.ReLU())
            

    def forward(self, x, record_state=False):
        if not self.see_double:
            return self.forward_single(x, record_state=record_state)
        else:
            out_1 = self.forward_single(x[:, :self.input_shape[0]], record_state=record_state)
            out_2 = self.forward_single(x[:, self.input_shape[0]:])            
            return torch.concat([out_1, out_2], dim=1)

    def forward_single(self, x, record_state=False):
        """encode the state or model's encoding inside the actor network
        args:
            x: input tensor of shape (B, C, H, W); can be state or model's encoding
        return:
            output: output tensor of shape (B, self.out_size)"""
        assert x.dtype in [torch.float, torch.float16]
        if not self.oned_input:
            if self.firstpool:
                x = self.down_pool_conv(x)
            if record_state: self.hidden_state = []
            for i, fconv in enumerate(self.feat_convs):                
                x = fconv(x)
                res_input = x
                x = self.resnet1[i](x)
                x += res_input
                res_input = x
                x = self.resnet2[i](x)
                x += res_input
                if record_state: self.hidden_state.append(x)
            x = torch.flatten(x, start_dim=1)
        else:
            x = self.input_block(x)
            x = self.res(x)
        x = self.fc(F.relu(x))
        if record_state: self.hidden_state = tile_and_concat_tensors(self.hidden_state)
        return x

class RNNEncoder(nn.Module):
    # RNN processor for 1d inputs; can be used directly on tree rep or encoded 3d input
    def __init__(self, 
                 in_size, # int; input size
                 flags            
                 ):
        super(RNNEncoder, self).__init__()  
        self.rnn_in_fc = nn.Sequential(
                    nn.Linear(in_size, flags.tran_dim), nn.ReLU()
        )  
        self.tran_layer_n = flags.tran_layer_n 
        if self.tran_layer_n > 0:
            self.rnn = ConvAttnLSTM(
                input_dim=flags.tran_dim,
                hidden_dim=flags.tran_dim,
                num_layers=flags.tran_layer_n,
                attn=not flags.tran_lstm_no_attn,
                mem_n=flags.tran_mem_n,
                num_heads=flags.tran_head_n,
                attn_mask_b=flags.tran_attn_b,
                tran_t=flags.tran_t,
            ) 
        self.rnn_out_fc = nn.Sequential(
            nn.Linear(flags.tran_dim, flags.tran_dim), nn.ReLU()
        )

    def initial_state(self, batch_size=1, device=None):
        if self.tran_layer_n > 0:
            return self.rnn.initial_state(batch_size, device=device)
        else:
            return ()

    def forward(self, x, done, core_state, record_state=False):
        # input should have shape (T*B, C) 
        # done should have shape (T, B)
        T, B = done.shape
        x = self.rnn_in_fc(x)
        if self.tran_layer_n >= 1:
            x = x.view(*((T, B) + x.shape[1:])).unsqueeze(-1).unsqueeze(-1)            
            core_output, core_state = self.rnn(x, done, core_state, record_state)
            core_output = torch.flatten(core_output, 0, 1)
            d = torch.flatten(core_output, 1)   
        else:
            d = x     
        d = self.rnn_out_fc(d)
        return d, core_state
    
class ActorBaseNet(nn.Module):
    # base class for all actor network
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=False, record_state=False):
        super(ActorBaseNet, self).__init__()
        self.disable_thinker = flags.wrapper_type == 1
        self.record_state = record_state        

        self.obs_space = obs_space        
        if not self.disable_thinker:
            self.pri_action_space = action_space[0][0]            
        else:
            self.pri_action_space = action_space[0]

        self.flags = flags      
        self.tree_rep_meaning = tree_rep_meaning

        self.float16 = flags.float16
        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.num_rewards += int(flags.cur_cost > 0.0)
        self.enc_type = flags.critic_enc_type  
        self.rv_tran = None
        self.critic_zero_init = flags.critic_zero_init         
        self.legacy = getattr(flags, "legacy", False)  

        # action space processing
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)
        
        self.ordinal = flags.actor_ordinal
        if self.ordinal:
            indices = torch.arange(self.num_actions).view(-1, 1)
            ordinal_mask = (indices + indices.T) <= (self.num_actions - 1)
            ordinal_mask = ordinal_mask.float()
            self.register_buffer("ordinal_mask", ordinal_mask)

        # state space processing
        self.see_tree_rep = flags.see_tree_rep and not self.disable_thinker
        if self.see_tree_rep:
            self.tree_reps_shape = obs_space["tree_reps"].shape[1:]             
            if self.legacy:
                self.tree_reps_shape = list(self.tree_reps_shape)
                self.tree_reps_shape[0] -= 2

        self.see_h = flags.see_h and not self.disable_thinker
        if self.see_h:
            self.hs_shape = obs_space["hs"].shape[1:]
        self.see_x = flags.see_x
        if self.see_x and not self.disable_thinker:
            self.xs_shape = obs_space["xs"].shape[1:]
        self.see_real_state = flags.see_real_state        
        
        if flags.see_real_state:
            assert obs_space["real_states"].dtype in ['uint8', 'float32'], f"Unupported observation sapce {obs_space['real_states']}"            
            low = torch.tensor(obs_space["real_states"].low[0])
            high = torch.tensor(obs_space["real_states"].high[0])
            self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()            
            if self.need_norm:
                self.register_buffer("norm_low", low)
                self.register_buffer("norm_high", high)
            self.real_states_shape = obs_space["real_states"].shape[1:]     

        if getattr(flags, "ppo_k", 1) > 1:
            kl_beta = torch.tensor(1.)
            self.register_buffer("kl_beta", kl_beta)

    def normalize(self, x):
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        if self.need_norm:
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
    def ordinal_encode(self, logits):
        norm_softm = F.sigmoid(logits)
        norm_softm_tiled = torch.tile(norm_softm.unsqueeze(-1), [1,1,1,self.num_actions])
        return torch.sum(torch.log(norm_softm_tiled + 1e-8) * self.ordinal_mask + torch.log(1 - norm_softm_tiled + 1e-8) * (1 - self.ordinal_mask), dim=-1)

    def get_weights(self):
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}    

    def set_weights(self, weights, strict=True):
        device = next(self.parameters()).device
        tensor = isinstance(next(iter(weights.values())), torch.Tensor)
        if not tensor:
            self.load_state_dict(
                {k: torch.tensor(v, device=device) for k, v in weights.items()}, strict=strict
            )
        else:
            self.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=strict)

class ActorNetSep(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(ActorNetSep, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        self.actor = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=True, critic=False)
        self.critic = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=False, critic=True)
        self.initial_state(1)
        self.rv_tran = self.critic.rv_tran

    def initial_state(self, batch_size, device=None):
        actor_state = self.actor.initial_state(batch_size, device)
        critic_state = self.critic.initial_state(batch_size, device)
        self.state_idx = len(actor_state)
        return actor_state + critic_state
    
    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        actor_state = core_state[:self.state_idx]
        critic_state = core_state[self.state_idx:]
        actor_out, actor_state = self.actor(env_out, actor_state, clamp_action, compute_loss, greedy)
        critic_out, critic_state = self.critic(env_out, critic_state, clamp_action, compute_loss, greedy)
        misc = actor_out.misc
        actor_out = ActorOut(
            pri=actor_out.pri,
            pri_param=actor_out.pri_param,
            reset=actor_out.reset,
            reset_logits=actor_out.reset_logits,
            action=actor_out.action,
            action_prob=actor_out.action_prob,
            c_action_log_prob=actor_out.c_action_log_prob,            
            baseline=critic_out.baseline,
            baseline_enc=critic_out.baseline_enc,
            entropy_loss=actor_out.entropy_loss,
            reg_loss=actor_out.reg_loss,
            misc=misc,
        )
        core_state = actor_state + critic_state
        return actor_out, core_state

class ActorNetSingle(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False, actor=True, critic=True):
        super(ActorNetSingle, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)      
                  
        self.actor = actor
        self.critic = critic

        if not self.discrete_action and self.actor:
            min_log_var = 2 * torch.log(torch.tensor(flags.actor_min_std))
            max_log_var = 2 * torch.log(torch.tensor(flags.actor_max_std))
            self.register_buffer("min_log_var", min_log_var)
            self.register_buffer("max_log_var", max_log_var)   

        self.tran_dim = flags.tran_dim 
        self.tree_rep_rnn = flags.tree_rep_rnn and flags.see_tree_rep         
        self.se_lstm_table = getattr(flags, "se_lstm_table", False) and flags.see_tree_rep and flags.wrapper_type in [3, 4]
        self.x_rnn = flags.x_rnn and flags.see_x  
        self.h_rnn = flags.h_rnn and flags.see_h
        self.real_state_rnn = flags.real_state_rnn and flags.see_real_state 

        self.sep_im_head = flags.sep_im_head
        self.last_layer_n = flags.last_layer_n
          
        # encoder for state or encoding output
        last_out_size = self.dim_rep_actions + self.num_rewards
        if self.legacy: last_out_size += self.dim_rep_actions

        if not self.disable_thinker:
            last_out_size += 2

        if self.see_h:
            FrameEncoder = AFrameEncoder if not self.legacy else AFrameEncoderLegacy
            self.h_encoder = FrameEncoder(
                input_shape=self.hs_shape,                    
                flags=flags,                      
            )
            h_out_size = self.h_encoder.out_size
            if self.h_rnn:
                rnn_in_size = h_out_size
                self.h_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                h_out_size = flags.tran_dim            
            last_out_size += h_out_size   
        
        if self.see_x:
            FrameEncoder = AFrameEncoder if not self.legacy else AFrameEncoderLegacy
            self.x_encoder_pre = FrameEncoder(
                input_shape=self.xs_shape,                 
                flags=flags,
                downpool=True,
                firstpool=True,
            )
            x_out_size = self.x_encoder_pre.out_size
            if self.x_rnn:
                rnn_in_size = x_out_size
                self.x_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                x_out_size = flags.tran_dim            
            last_out_size += x_out_size           

        if self.see_real_state:
            self.real_state_encoder =  AFrameEncoder(
                input_shape=self.real_states_shape,                 
                flags=flags,
                downpool=True,
                firstpool=True,
            )
            r_out_size = self.real_state_encoder.out_size
            if self.real_state_rnn:
                rnn_in_size = r_out_size
                self.r_encoder_rnn = RNNEncoder(
                    in_size=rnn_in_size,
                    flags=flags,
                )
                r_out_size = flags.tran_dim   
            last_out_size += r_out_size       

        if self.see_tree_rep:            
            self.tree_rep_meaning = tree_rep_meaning
            in_size = self.tree_reps_shape[0]
            if self.se_lstm_table:
                assert flags.se_query_cur == 2                
                root_table_mask = torch.zeros(in_size, dtype=torch.bool)
                root_query_keys = [k for k in tree_rep_meaning if k.startswith("root_query")]
                for i in root_query_keys:
                    root_table_mask[self.tree_rep_meaning[i]] = 1        
                # print("root_query_size: ", sum(root_table_mask).long().item())        
                cur_table_mask = torch.zeros(in_size, dtype=torch.bool)
                cur_query_keys = [k for k in tree_rep_meaning if k.startswith("cur_query")]
                for i in cur_query_keys:
                    cur_table_mask[self.tree_rep_meaning[i]] = 1
                # print("cur_query_size: ", sum(cur_table_mask).long().item())        
                non_table_mask = torch.logical_or(root_table_mask, cur_table_mask)
                non_table_mask = torch.logical_not(non_table_mask)
                self.register_buffer("root_table_mask", root_table_mask)
                self.register_buffer("cur_table_mask", cur_table_mask)
                self.register_buffer("non_table_mask", non_table_mask)
                input_size = (sum(root_table_mask) / flags.se_query_size).long().item()
                self.tree_rep_table_lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=3, batch_first=True)
                in_size = torch.sum(non_table_mask).long() + 64 * 2           

            if self.tree_rep_rnn:
                self.tree_rep_encoder = RNNEncoder(
                    in_size=in_size,
                    flags=flags
                )
                last_out_size += flags.tran_dim
            else:
                self.tree_rep_encoder = MLP(
                    input_size=in_size,
                    layer_sizes=[200, 200, 200],
                    output_size=100,
                    norm=False,
                    skip_connection=True,
                )
                last_out_size += 100        

        if self.last_layer_n > 0:
            self.final_mlp =  MLP(
                input_size=last_out_size,
                layer_sizes=[200]*self.last_layer_n,
                output_size=100,
                norm=False,
                skip_connection=True,
            )
            last_out_size = 100

        if self.actor:
            self.policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
            self.im_policy = self.policy
            if not self.discrete_action:
                self.tanh_action = flags.tanh_action
                self.policy_lvar = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                self.im_policy_lvar = self.policy

            if not self.disable_thinker:
                if self.sep_im_head:
                    self.im_policy = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                    if not self.discrete_action:
                        self.im_policy_lvar = nn.Linear(last_out_size, self.num_actions * self.dim_actions)
                    
                self.reset = nn.Linear(last_out_size, 2)

        if self.critic:
            self.rv_tran = None
            if self.enc_type == 0:
                self.baseline = nn.Linear(last_out_size, self.num_rewards)
                if self.flags.reward_clip > 0:
                    self.baseline_clamp = self.flags.reward_clip / (
                        1 - self.flags.discounting
                    )
            elif self.enc_type == 1:
                self.baseline = nn.Linear(last_out_size, self.num_rewards)
                self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
            elif self.enc_type in [2, 3]:                        
                self.rv_tran = RVTran(enc_type=self.enc_type, enc_f_type=flags.critic_enc_f_type)
                self.out_n = self.rv_tran.encoded_n
                self.baseline = nn.Linear(last_out_size, self.num_rewards * self.out_n)            

            if self.critic_zero_init:
                nn.init.constant_(self.baseline.weight, 0.0)
                nn.init.constant_(self.baseline.bias, 0.0)                

        self.initial_state(batch_size=1) # initialize self.state_idx        

    def initial_state(self, batch_size, device=None):
        self.state_idx = {}
        idx = 0
        initial_state = ()
        
        conditions = [self.x_rnn, self.real_state_rnn, self.tree_rep_rnn, self.h_rnn]
        rnn_names = ["x_encoder_rnn", "r_encoder_rnn", "tree_rep_encoder", "h_encoder_rnn"]
        state_names = ["x", "r", "tree_rep", "h"]

        for condition, rnn_name, state_name in zip(conditions, rnn_names, state_names):
            if condition:
                core_state = getattr(self, rnn_name).initial_state(batch_size, device=device)
                initial_state = initial_state + core_state
                self.state_idx[state_name] = slice(idx, idx+len(core_state))
                idx += len(core_state)

        self.state_len = idx
        return initial_state
    
    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        """one-step forward for the actor;
        args:
            env_out (EnvOut):
                tree_reps (tensor): tree_reps output with shape (T x B x C)
                xs (tensor): optional - model predicted state with shape (T x B x C X H X W)                
                hs (tensor): optional - hidden state with shape (T x B x C X H X W)                
                real_states (tensor): optional - root's real state with shape (T x B x C X H X W)                
                done  (tensor): if episode ends with shape (T x B)
                step_status (tensor): current step status with shape (T x B)
                last_pri (tensor): last primiary action (non-one-hot) with shape (T x B)
                last_reset (tensor): last reset action (non-one-hot) with shape (T x B)
                and other environment output that is not used.
            core_state (tuple): rnn state of the actor network
            clamp_action (tuple): option - if not none, the sampled action will be set to this action;
                the main purpose is for computing c_action_log_prob
            compute_loss (boolean): wheather to return entropy loss and reg loss
            greedy (bool): whether to sample greedily
        return:
            ActorOut:
                see definition of ActorOut; this is a tuple with elements of 
                    shape (T x B x ...) except actor_out.action, which is a 
                    tuple of primiary and reset action, each with shape (B,),
                    selected on the last step
        """
        done = env_out.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape

        assert len(core_state) == self.state_len, "core_state should have length %d" % self.state_len
        new_core_state = [None] * self.state_len

        final_out = []
        
        last_pri = torch.flatten(env_out.last_pri, 0, 1)
        if not self.tuple_action: last_pri = last_pri.unsqueeze(-1)
        last_pri = util.encode_action(last_pri, self.pri_action_space)   
        final_out.append(last_pri)
        if self.legacy:
            final_out.append(last_pri)

        if not self.disable_thinker:
            last_reset = torch.flatten(env_out.last_reset, 0, 1)
            last_reset = F.one_hot(last_reset, 2)
            final_out.append(last_reset)

        reward = env_out.reward
        reward[torch.isnan(reward)] = 0.
        last_reward = torch.clamp(torch.flatten(reward, 0, 1), -1, +1)
        final_out.append(last_reward)

        if self.see_tree_rep:                
            tree_rep = env_out.tree_reps               
            tree_rep = torch.flatten(tree_rep, 0, 1)     
            if self.legacy:
                indices_to_remove = [self.num_actions+1, self.num_actions*5+6 + self.num_actions+1]
                mask = torch.ones(tree_rep.shape[1], dtype=torch.bool, device=tree_rep.device)
                mask[indices_to_remove] = False
                tree_rep = tree_rep[:, mask]

            if self.se_lstm_table:
                root_table = tree_rep[:, self.root_table_mask]
                root_table = torch.flip(root_table.view(T*B, self.flags.se_query_size, -1), dims=[1])
                root_table_rep, _ = self.tree_rep_table_lstm(root_table)
                root_table_rep = root_table_rep[:, -1]
                cur_table = tree_rep[:, self.cur_table_mask]
                cur_table = torch.flip(cur_table.view(T*B, self.flags.se_query_size, -1), dims=[1])
                cur_table_rep, _ = self.tree_rep_table_lstm(cur_table)
                cur_table_rep = cur_table_rep[:, -1]
                tree_rep = torch.concat([tree_rep[:, self.non_table_mask], root_table_rep, cur_table_rep], dim=-1)

            if self.tree_rep_rnn:
                core_state_ = core_state[self.state_idx['tree_rep']]
                encoded_tree_rep, core_state_ = self.tree_rep_encoder(
                    tree_rep, done, core_state_)
                new_core_state[self.state_idx['tree_rep']] = core_state_
            else:
                encoded_tree_rep = self.tree_rep_encoder(tree_rep)
            final_out.append(encoded_tree_rep)
        
        if self.see_h:
            hs = torch.flatten(env_out.hs, 0, 1)
            encoded_h = self.h_encoder(hs)            
            if self.legacy and self.see_tree_rep:
                final_out[-1], final_out[-2] = final_out[-2], final_out[-1]

            if self.h_rnn:
                core_state_ = core_state[self.state_idx['h']]
                encoded_h, core_state_ = self.h_encoder_rnn(
                    encoded_h, done, core_state_)
                new_core_state[self.state_idx['h']] = core_state_

            final_out.append(encoded_h)                

        if self.see_x:
            xs = torch.flatten(env_out.xs, 0, 1)
            with autocast(enabled=self.float16):                
                encoded_x = self.x_encoder_pre(xs)
            if self.float16: encoded_x = encoded_x.float()
                
            if self.x_rnn:
                core_state_ = core_state[self.state_idx['x']]
                encoded_x, core_state_ = self.x_encoder_rnn(
                    encoded_x, done, core_state_)
                new_core_state[self.state_idx['x']] = core_state_
            
            final_out.append(encoded_x)

        if self.see_real_state:
            real_states = torch.flatten(env_out.real_states, 0, 1)   
            real_states = self.normalize(real_states.float())
            with autocast(enabled=self.float16):      
                encoded_real_state = self.real_state_encoder(real_states, record_state=self.record_state)
            if self.float16: encoded_real_state = encoded_real_state.float()

            if self.real_state_rnn:
                core_state_ = core_state[self.state_idx['r']]
                encoded_real_state, core_state_ = self.r_encoder_rnn(
                    encoded_real_state, done, core_state_, record_state=self.record_state)
                new_core_state[self.state_idx['r']] = core_state_
                if self.record_state: self.hidden_state = self.r_encoder_rnn.rnn.hidden_state
            else:
                if self.record_state: self.hidden_state = self.real_state_encoder.hidden_state

            final_out.append(encoded_real_state)

        final_out = torch.concat(final_out, dim=-1)   

        if self.last_layer_n > 0:
            final_out = self.final_mlp(final_out)     

        misc = {}
        if self.actor:
            # compute logits
            pri_logits = self.policy(final_out)    
            pri_logits = pri_logits.view(T*B, self.dim_actions, self.num_actions)
            if self.ordinal: pri_logits = self.ordinal_encode(pri_logits)
            if not self.discrete_action:
                pri_mean = pri_logits[:, :, 0]
                pri_log_var = self.policy_lvar(final_out)

            if not self.discrete_action:
                pri_log_var = torch.clamp(pri_log_var, self.min_log_var, self.max_log_var)

            if not self.disable_thinker:
                im_logits = self.im_policy(final_out)                
                im_logits = im_logits.view(T*B, self.dim_actions, self.num_actions)
                if self.ordinal: im_logits = self.ordinal_encode(im_logits)
                if not self.discrete_action:                        
                    im_mean = im_logits[:, :, 0]
                    im_log_var = self.im_policy_lvar(final_out) 
                if not self.discrete_action:                   
                    im_log_var = torch.clamp(im_log_var, self.min_log_var, self.max_log_var)

                im_mask = env_out.step_status <= 1 # imagainary action will be taken next
                if self.discrete_action:
                    im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1).unsqueeze(-1)
                    pri_logits = torch.where(im_mask, im_logits, pri_logits)
                else:                    
                    im_mask = torch.flatten(im_mask, 0, 1).unsqueeze(-1)
                    pri_mean = torch.where(im_mask, im_mean, pri_mean)
                    pri_log_var = torch.where(im_mask, im_log_var, pri_log_var)
                reset_logits = self.reset(final_out)
            else:   
                reset_logits = None

            # compute entropy loss
            if compute_loss:
                if self.discrete_action:
                    entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                        input=torch.flatten(pri_logits, 0, 1), 
                        target=torch.flatten(F.softmax(pri_logits, dim=-1), 0, 1),                
                    )
                    entropy_loss = entropy_loss.view(T, B, self.dim_actions)
                    entropy_loss = torch.sum(entropy_loss, dim=-1)
                else:
                    entropy_loss = -torch.sum(pri_log_var.view(T, B, self.dim_actions), dim=-1)
                if not self.disable_thinker:
                    ent_reset_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                        input=reset_logits, target=F.softmax(reset_logits, dim=-1)
                    )
                    ent_reset_loss = ent_reset_loss.view(T, B) * (env_out.step_status <= 1).float()
                    entropy_loss = entropy_loss + ent_reset_loss 
            else:
                entropy_loss = None

            # sample action
            if self.discrete_action:
                pri = sample(pri_logits, greedy=greedy, dim=-1)
                pri_logits = pri_logits.view(T, B, self.dim_actions, self.num_actions)
                pri = pri.view(T, B, self.dim_actions)        
                pri_param = pri_logits
            else:
                pri_mean = pri_mean.view(T, B, self.dim_actions)
                pri_log_var = pri_log_var.view(T, B, self.dim_actions)                
                pri_std = torch.exp(pri_log_var / 2)
                pri_std = pri_std.view(T, B, self.dim_actions)
                normal_dist = torch.distributions.Normal(pri_mean, pri_std)
                if not greedy:                
                    pri_pre_tanh = normal_dist.sample()
                else:
                    pri_pre_tanh = pri_mean
                if self.tanh_action:
                    pri = torch.tanh(pri_pre_tanh)
                else:
                    pri = pri_pre_tanh
                pri_param = torch.stack((pri_mean, pri_log_var), dim=-1)

            if not self.disable_thinker:
                reset = sample(reset_logits, greedy=greedy, dim=-1)
                reset_logits = reset_logits.view(T, B, 2)
                reset = reset.view(T, B)    
            else:
                reset = None

            # clamp the action to clamp_action
            if clamp_action is not None:
                if not self.disable_thinker:
                    pri[:clamp_action[0].shape[0]] = clamp_action[0]
                    reset[:clamp_action[1].shape[0]] = clamp_action[1]
                else:
                    pri[:clamp_action.shape[0]] = clamp_action
                if not self.discrete_action:  
                    if self.tanh_action:              
                        pri_pre_tanh = atanh(pri)
                    else:
                        pri_pre_tanh = pri

            # compute chosen log porb
            if self.discrete_action:
                c_action_log_prob = compute_discrete_log_prob(pri_logits, pri)     
            else:
                c_action_log_prob = normal_dist.log_prob(pri_pre_tanh)
                if self.tanh_action:    
                    c_action_log_prob = c_action_log_prob - torch.log(1.0 - pri ** 2 + 1e-6)
                c_action_log_prob = torch.sum(c_action_log_prob, dim=-1)                

            if not self.disable_thinker:
                c_reset_log_prob = compute_discrete_log_prob(reset_logits, reset)     
                c_reset_log_prob = c_reset_log_prob * (env_out.step_status <= 1).float()
                # if next action is real action, reset will never be used
                c_action_log_prob += c_reset_log_prob

            # pack last step's action and action prob        
            pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]        
            if not self.disable_thinker:
                action = (pri_env, reset[-1])            
            else:
                action = pri_env        

            if self.discrete_action:   
                action_prob = F.softmax(pri_logits, dim=-1)    
            else:
                action_prob = pri_param
            if not self.tuple_action: action_prob = action_prob[:, :, 0]    

        if self.critic:
            # compute baseline
            if self.enc_type == 0:
                baseline = self.baseline(final_out)
                if self.flags.reward_clip > 0:
                    baseline = torch.clamp(
                        baseline, -self.baseline_clamp, +self.baseline_clamp
                    )
                baseline_enc = None
            elif self.enc_type == 1:
                baseline_enc_s = self.baseline(final_out)
                baseline = self.rv_tran.decode(baseline_enc_s)
                baseline_enc = baseline_enc_s
            elif self.enc_type in [2, 3]:
                baseline_enc_logit = self.baseline(final_out).reshape(
                    T * B, self.num_rewards, self.out_n
                )
                baseline_enc_v = F.softmax(baseline_enc_logit, dim=-1)
                baseline = self.rv_tran.decode(baseline_enc_v)
                baseline_enc = baseline_enc_logit

            baseline_enc = (
                baseline_enc.view((T, B) + baseline_enc.shape[1:])
                if baseline_enc is not None
                else None
            )
            baseline = baseline.view(T, B, self.num_rewards)

        if compute_loss:
            reg_loss = 1e-6 * torch.sum(final_out**2, dim=-1).view(T, B) / 2
            if self.discrete_action and self.actor:
                reg_loss += 1e-3 * torch.sum(pri_logits**2, dim=(-2,-1)) / 2
            if not self.disable_thinker and self.actor:
                reg_loss += (
                    + 1e-3 * torch.sum(reset_logits**2, dim=-1) / 2
                )
        else:
            reg_loss = None
        
        actor_out = ActorOut(
            pri=pri if self.actor else None,
            pri_param=pri_param if self.actor else None,
            reset=reset if self.actor else None,
            reset_logits=reset_logits if self.actor else None,
            action=action if self.actor else None,
            action_prob=action_prob if self.actor else None,
            c_action_log_prob=c_action_log_prob if self.actor else None,            
            baseline=baseline if self.critic else None,
            baseline_enc=baseline_enc if self.critic else None,
            entropy_loss=entropy_loss if self.actor else None,
            reg_loss=reg_loss,
            misc=misc,
        )
        core_state = tuple(new_core_state)
        return actor_out, core_state    

class DRCNet(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(DRCNet, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        assert flags.wrapper_type == 1

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_space["real_states"].shape[1], out_channels=32, kernel_size=8, stride=4, padding=2
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
        )
        output_shape = lambda h, w, kernel, stride, padding: (
            ((h + 2 * padding - kernel) // stride + 1),
            ((w + 2 * padding - kernel) // stride + 1),
        )

        h, w = output_shape(self.real_states_shape[1], self.real_states_shape[2], 8, 4, 2)
        h, w = output_shape(h, w, 4, 2, 1)

        self.core = ConvAttnLSTM(            
            input_dim=32,
            hidden_dim=32,
            num_layers=3,
            attn=False,
            h=h,
            w=w,            
            kernel_size=3,
            mem_n=None,            
            num_heads=8,            
            attn_mask_b=None,
            tran_t=3,
            pool_inject=True,
        )
        last_out_size = 32 * h * w * 2
        self.final_layer = nn.Linear(last_out_size, 256)
        self.policy = nn.Linear(256, self.num_actions * self.dim_actions)
        self.baseline = nn.Linear(256, 1)

        if getattr(flags, "ppo_k", 1) > 1:
            kl_beta = torch.tensor(1.)
            self.register_buffer("kl_beta", kl_beta)

    def initial_state(self, batch_size, device=None):
        return self.core.initial_state(batch_size, device=device)

    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        done = env_out.done
        assert (
            len(done.shape) == 2
        ), f"done shape should be (T, B) instead of {done.shape}"
        T, B = done.shape
        x = self.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.encoder(x)
        core_input = x_enc.view(*((T, B) + x_enc.shape[1:]))
        core_output, core_state = self.core(core_input, done, core_state, record_state=self.record_state)
        if self.record_state: self.hidden_state = self.core.hidden_state
        core_output = torch.flatten(core_output, 0, 1)
        core_output = torch.cat([x_enc, core_output], dim=1)
        core_output = torch.flatten(core_output, 1)
        final_out = F.relu(self.final_layer(core_output))

        pri_logits = self.policy(final_out)
        pri_logits = pri_logits.view(T*B, self.dim_actions, self.num_actions)

        # compute entropy loss
        if compute_loss:
            entropy_loss = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(pri_logits, 0, 1), 
                target=torch.flatten(F.softmax(pri_logits, dim=-1), 0, 1),
            )
            entropy_loss = entropy_loss.view(T, B, self.dim_actions)            
            entropy_loss = torch.sum(entropy_loss, dim=-1)
        else:
            entropy_loss = None

        # sample_action
        pri = sample(pri_logits, greedy=greedy, dim=-1)
        pri_logits = pri_logits.view(T, B, self.dim_actions, self.num_actions)
        pri = pri.view(T, B, self.dim_actions)      

        # clamp the action to clamp_action
        if clamp_action is not None:
            pri[:clamp_action.shape[0]] = clamp_action

        # compute chosen log porb
        c_action_log_prob = compute_discrete_log_prob(pri_logits, pri)    

        # pack last step's action and action prob        
        pri_env = pri[-1, :, 0] if not self.tuple_action else pri[-1]   
        action = pri_env  
        action_prob = F.softmax(pri_logits, dim=-1)
        if not self.tuple_action: action_prob = action_prob[:, :, 0]    

        baseline = self.baseline(final_out).view(T, B, 1)

        if compute_loss:
            reg_loss = (
                1e-3 * torch.sum(torch.square(pri_logits), dim=(-2, -1))
                + 1e-5 * torch.sum(torch.square(self.baseline.weight)) 
                + 1e-5 * torch.sum(torch.square(self.policy.weight))
            )
        else:
            reg_loss = None
            
        actor_out = ActorOut(
            pri=pri,
            pri_param=pri_logits,
            reset=None,
            reset_logits=None,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=c_action_log_prob,            
            baseline=baseline,
            baseline_enc=None,
            entropy_loss=entropy_loss,
            reg_loss=reg_loss,
            misc={},
        )
        return actor_out, core_state

class MCTS(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(MCTS, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        assert flags.wrapper_type in [0, 2], "MCTS only support wrapper_type 0, 2"
        assert not flags.tree_carry, "MCTS does not support tree carry"
        assert type(action_space[0][0]) == spaces.discrete.Discrete, f"Unsupported action space f{action_space}"
        
        self.temp = 1
        self.dir_dist = None
        self.root_psa = None            

    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        tree_rep = env_out.tree_reps  
        T, B, C = tree_rep.shape
        assert T == 1
        tree_rep = tree_rep[0]

        assert torch.all(env_out.step_status == env_out.step_status[0, 0]), f"step_status should be the same for all item, not {env_out.step_status}."
        step_status = env_out.step_status[0, 0]
        last_real_step = step_status in [0, 3]
        next_real_step = step_status in [2, 3]                
        
        if last_real_step:
            # last step is real, re init. variables   
            root_logits = torch.clone(tree_rep[:, self.tree_rep_meaning["root_policy"]])  
            self.root_psa = F.softmax(root_logits, dim=-1)
            if self.dir_dist is None:
                con = torch.tensor([0.3]*self.num_actions, device=tree_rep.device)
                self.dir_dist = torch.distributions.dirichlet.Dirichlet(con, validate_args=None)
            self.dir_noise = self.dir_dist.sample((B,))
            self.root_psa = self.root_psa * 0.75 + self.dir_noise * 0.25

        if next_real_step:
            # real step
            root_nsa = tree_rep[:, self.tree_rep_meaning["root_ns"]] * self.flags.rec_t            
            if not greedy:
                root_nsa_temp = root_nsa ** (1 / self.temp)
                pri_prob = root_nsa_temp / torch.sum(root_nsa_temp, dim=-1, keepdim=True)
                pri = torch.multinomial(pri_prob, num_samples=1)[:, 0]
            else:
                pri = torch.argmax(root_nsa, dim=-1)
                pri_prob = F.one_hot(pri, self.num_actions)  

            reset = torch.ones_like(pri)      
        else:
            # imaginary step            
            reset_m = tree_rep[:, self.tree_rep_meaning["cur_reset"]].squeeze(-1) == 1

            if last_real_step:
                cur_psa = self.root_psa
            else:
                cur_logits = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_policy"]])  
                cur_psa = F.softmax(cur_logits, dim=-1)
                if self.root_psa is not None:
                    cur_psa[reset_m] = self.root_psa[reset_m]
                else:
                    print("Warning: root_psa is not initialized. Make sure the first state has step_status 0 or 3")

            cur_nsa = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_ns"]])    
            root_nsa = torch.clone(tree_rep[:, self.tree_rep_meaning["root_ns"]])    
            cur_nsa[reset_m] = root_nsa[reset_m]
            cur_nsa = cur_nsa * self.flags.rec_t
            
            # compute normalized q(s,a)
            cur_qsa = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_qs_mean"]])    
            root_qsa = torch.clone(tree_rep[:, self.tree_rep_meaning["root_qs_mean"]])    
            cur_qsa[reset_m] = root_qsa[reset_m]

            # normalization (see https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py#L87)
            cur_v = torch.clone(tree_rep[:, self.tree_rep_meaning["cur_v"]])    
            root_v = torch.clone(tree_rep[:, self.tree_rep_meaning["root_v"]])    
            cur_v[reset_m] = root_v[reset_m]

            cur_qsa[cur_nsa==0] = cur_v.broadcast_to(B, self.num_actions)[cur_nsa==0]
            q_min = torch.minimum(cur_v.squeeze(-1), torch.min(cur_qsa, dim=-1)[0])
            q_max = torch.maximum(cur_v.squeeze(-1), torch.max(cur_qsa, dim=-1)[0])            
            cur_qsa = (cur_qsa - q_min.unsqueeze(-1)) / (q_max.unsqueeze(-1) - q_min.unsqueeze(-1) + 1e-8)
            cur_qsa[cur_nsa==0] = 0.

            assert torch.all((cur_qsa >= 0) & (cur_qsa <= 1)), f"normalized cur_qsa should range from [0, 1], not {cur_qsa}"

            c_1 = 1.25
            c_2 = 19652
            sum_cur_nsa = torch.sum(cur_nsa, dim=-1, keepdim=True)
            score = cur_qsa + cur_psa * (torch.sqrt(sum_cur_nsa)) / (1 + cur_nsa) * (
                c_1 + torch.log((sum_cur_nsa + c_2 + 1) / c_2)
            )
            pri = torch.argmax(score, dim=-1)
            pri_prob = F.one_hot(pri, self.num_actions)      

            reset = (torch.sum(cur_nsa, dim=-1) <= 0).long()
        
        pri = pri.view(T, B, 1)
        reset = reset.view(T, B)
        action = (pri[-1, :, 0], reset[-1])  
        action_prob = pri_prob.view(T, B, self.num_actions)

        actor_out = ActorOut(
            pri=pri,
            pri_param=None,
            reset=reset,
            reset_logits=None,
            action=action,
            action_prob=action_prob,
            c_action_log_prob=None,            
            baseline=None,
            baseline_enc=None,
            entropy_loss=None,
            reg_loss=None,
            misc={},
        )
        return actor_out, core_state    
    
    def set_real_step(self, real_step):
        if real_step < self.flags.total_steps * 0.5:
            self.temp = 1
        elif real_step < self.flags.total_steps * 0.75:
            self.temp = 0.5
        else:
            self.temp = 0.25
    
    def initial_state(self, batch_size, device=None):
        return ()
    
    def set_weights(self, weights):
        return
    
    def get_weights(self):
        return {}
    
    def to(self, device):
        return self
    
    def train(self, train):
        return     
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def ActorNet(*args, **kwargs):

    if getattr(kwargs["flags"], "drc", False):        
        Net = DRCNet
    elif getattr(kwargs["flags"], "mcts", False):  
        Net = MCTS
    elif not getattr(kwargs["flags"], "sep_actor_critic", False):
        Net = ActorNetSingle
    else:
        Net = ActorNetSep

    return Net(*args, **kwargs)