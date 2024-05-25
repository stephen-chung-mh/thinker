import os
import numpy as np
import time
import timeit
import traceback
import ray
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from thinker.core.file_writer import FileWriter
from thinker.core.module import guassian_kl_div
from thinker.model_net import ModelNet, VPNet
import thinker.util as util
import gc
from collections import namedtuple

def compute_cross_entropy_loss(policy, target_policy, discrete_action, require_prob, is_weights, mask=None):
    k, b, d, _ = policy.shape
    if discrete_action:
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(policy, 0, 2), target=torch.flatten(target_policy, 0, 2)
        )
        loss = loss.view(k, b, d)
        loss = torch.mean(loss, dim=2)
    elif require_prob:
        tar_mean = target_policy[:, :, :, 0]
        tar_log_var = target_policy[:, :, :, 1]
        mean = policy[:, :, :, 0]
        log_var = policy[:, :, :, 1]
        loss = guassian_kl_div(
            tar_mean, tar_log_var, mean, log_var, reduce="mean"
        )
    else:
        loss = 0.5 * (log_var + ((policy - mean) ** 2) /  torch.exp(log_var))
        loss = torch.mean(loss, dim=-1)
    if mask is not None: loss = loss * mask
    loss = torch.sum(loss, dim=0)
    loss = is_weights * loss
    return torch.sum(loss)
   
class SModelLearner:
    def __init__(self, name, ray_obj, model_param, flags, model_net=None, device=None):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()

        if flags.parallel:
            self.model_buffer = ray_obj["model_buffer"]
            self.param_buffer = ray_obj["param_buffer"]
            self.signal_buffer = ray_obj["signal_buffer"]
            self.model_net = ModelNet(**model_param)
            self.refresh_model()
            self.model_net.train(True)
            if self.flags.gpu_learn > 0. and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:           
                self.device = torch.device("cpu")
        else:
            assert model_net is not None, "actor_net is required for non-parallel mode"
            assert device is not None, "device is required for non-parallel mode"
            self.model_net = model_net
            self.device = device

        if self.device == torch.device("cuda"):
            self._logger.info("Init. model-learning: Using CUDA.")
        else:
            self._logger.info("Init. model-learning: Not using CUDA.")

        self.step = 0
        self.real_step = 0

        lr_lambda = (
            lambda epoch: 1
            - min(epoch, self.flags.total_steps) / self.flags.total_steps
        )

        opt = getattr(flags, "model_optimizer", "adam")
        if opt == "adam":
            Optimizer = torch.optim.Adam
            opt_args = {}
        elif opt == "sgd":
            Optimizer = torch.optim.SGD
            opt_args = {
                "momentum": self.flags.model_sgd_momentum,
                "weight_decay": self.flags.model_sgd_weight_decay,
            }

        if self.flags.dual_net:
            self.optimizer_m = Optimizer(
                self.model_net.sr_net.parameters(), lr=flags.model_learning_rate, **opt_args
            )
            self.scheduler_m = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_m, lr_lambda
            )
            self.scaler_m = GradScaler(init_scale=2**3) if self.flags.float16 else None
        
        param_groups = self.model_net.vp_net.parameters()
        self.optimizer_p = Optimizer(param_groups, lr=flags.model_learning_rate, **opt_args)

        self.scheduler_p = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_p, lr_lambda
        )
        self.scaler_p = GradScaler(init_scale=2**3) if self.flags.float16 else None

        self.ckp_path = os.path.join(flags.ckpdir, "ckp_model.tar")
        if flags.ckp: self.load_checkpoint(self.ckp_path)

        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            suffix="_model",
            overwrite=not self.flags.ckp,
        )

        # move network and optimizer to process device
        self.model_net.to(self.device)
        if self.flags.dual_net:
            util.optimizer_to(self.optimizer_m, self.device)
        util.optimizer_to(self.optimizer_p, self.device)               

        self.timing = util.Timings() if self.time else None
        self.perfect_model = util.check_perfect_model(flags.wrapper_type)

        # other init. variables for consume_data
        self.last_psteps = 0
        self.timer = timeit.default_timer
        self.start_step = self.step
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.sps_buffer_n = 0
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.n = 0

        self.model_T = flags.model_unroll_len + 1
        self.model_B = flags.model_batch_size
        self.numel_per_step = self.model_T * self.model_B
        self.replay_ratio = 0

        if flags.parallel:
            self.data_ptr = self.read_buffer_ptr()
        self.start_training = False
        self.finish = False

    def read_buffer_ptr(self):
        return self.model_buffer.read.remote(self.model_T, self.model_B, self.compute_beta(), add_t=self.flags.model_return_n+1)

    def compute_beta(self):
        c = min(self.real_step, self.flags.total_steps) / self.flags.total_steps
        return self.flags.priority_beta * (1 - c) + 1.0 * c
    
    def init_psteps(self, data):
        if data is not None and not self.start_training:                                    
            # record the last processed steps from buffer
            self.last_psteps = int(data["processed_n"])
            if not self.flags.ckp:
                self.real_step += self.last_psteps
                # if it is not loading from checkpoint, the steps
                # used to fill the model should also be counted
            self.start_training = True   
    
    def log_preload(self, status):
        if self.timer() - self.start_time > 5:
            self._logger.info(
                "[%s] Preloading: %d/%d"
                % (self.flags.xpid, status["processed_n"], status["warm_up_n"])
            )
            self.start_time = self.timer()

    def learn_data(self):
        try:
            data_ptr = self.read_buffer_ptr()

            while self.real_step < self.flags.total_steps:
                if self.time: self.timing.reset()
                # get data remotely
                if self.replay_ratio < self.flags.max_replay_ratio:
                    while True:                    
                        data = ray.get(data_ptr)
                        ray.internal.free(data_ptr)
                        data_ptr = self.read_buffer_ptr()
                        self.init_psteps(data)
                        if data is not None: break
                        time.sleep(0.01)
                        status = ray.get(self.model_buffer.get_status.remote())
                        self.log_preload(status)                    
                        if status["finish"]: 
                            self.finish = True
                            break                    

                    if self.time: self.timing.time("get_data")
                    if data == "FINISH" or self.finish: break
                    self.replay_ratio = data["replay_ratio"]

                    # start consume data
                    self.consume_data(data)
                    del data                
                    gc.collect()
                    model_update = True
                else:
                    model_update = False

                # update shared buffer's weights
                if model_update:
                    self.param_buffer.set_data.remote(
                        "model_net", self.model_net.get_weights()
                    )
                if self.time: self.timing.time("update_weight")

                if int(time.strftime("%M")) // 10 != self.ckp_start_time:
                    self.save_checkpoint()
                    self.ckp_start_time = int(time.strftime("%M")) // 10
                if self.timing is not None:
                    self.timing.time("misc")

                if not model_update:
                    time.sleep(0.01)
                    status = ray.get(self.model_buffer.get_status.remote())
                    self.replay_ratio = status["replay_ratio"]
                    if status["finish"]: 
                        self.finish = True
                        break 

            self._logger.info("Terminating model-learning thread")
            self.model_buffer.set_finish.remote()
            self.signal_buffer.update_dict_item.remote(
                "self_play_signals", "halt", False
            )
            self.close()
            return True

        except Exception as e:
            self._logger.error(f"Exception detected in learn_model: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close()
            return True
        
    def update_real_step(self, data):
        new_psteps = data["processed_n"]
        new_psteps = int(new_psteps)        
        self.real_step += new_psteps - self.last_psteps
        self.last_psteps = new_psteps

    def consume_data(self, data, model_buffer=None):
        # model_buffer is only provided in non-parallel mode
        # which is required for updating the priorities of 
        # transition in the buffer
        self.n += 1
        self.update_real_step(data)
        train_model_out, is_weights, idx = data["data"], data["weights"], data["idx"]
        TrainModelOut = namedtuple('TrainModelOut', train_model_out.keys())
        train_model_out = TrainModelOut(**train_model_out)
        # move the data to the process device to free memory
        train_model_out = util.tuple_map(
            train_model_out, lambda x: torch.tensor(x, device=self.device)
        )
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        del data

        target = self.prepare_data(train_model_out)
        if self.timing is not None:
            self.timing.time("convert_data")

        if self.flags.dual_net:
            torch.autograd.set_detect_anomaly(True)
            # compute losses for model_net
            with autocast(enabled=self.flags.float16):
                losses_m, pred_xs = self.compute_losses_m(
                    train_model_out, target, is_weights
                )
            if self.timing is not None:
                self.timing.time("compute_losses_m")
            total_norm_m = self.gradient_step(
                losses_m["total_loss_m"], self.optimizer_m, self.scheduler_m, self.scaler_m
            )
            if self.timing is not None:
                self.timing.time("gradient_step_m")            
        else:
            losses_m = {}
            total_norm_m = torch.zeros(1, device=self.device)
            pred_xs = None
        with autocast(enabled=self.flags.float16):
            losses_p, priorities = self.compute_losses_p(
                train_model_out, target, is_weights, pred_xs
            )
        if self.timing is not None:
            self.timing.time("compute_losses_p")
        total_norm_p = self.gradient_step(
            losses_p["total_loss_p"], self.optimizer_p, self.scheduler_p, self.scaler_p
        )
        if self.timing is not None:
            self.timing.time("gradient_step_p")
        if self.flags.priority_alpha > 0:
            if model_buffer is None:
                self.model_buffer.update_priority.remote(idx, priorities)
            else:
                model_buffer.update_priority(idx, priorities)
        self.step += self.numel_per_step
        if self.timing is not None:
            self.timing.time("update_priority")
        losses = losses_m
        losses.update(losses_p)
        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            sps = (
                self.sps_buffer[self.sps_buffer_n - 1][0]
                - self.sps_buffer[self.sps_buffer_n][0]
            ) / (
                self.sps_buffer[self.sps_buffer_n - 1][1]
                - self.sps_buffer[self.sps_buffer_n][1]
            )
            tot_sps = (self.step - self.sps_start_step) / (
                self.timer() - self.sps_start_time
            )
            print_str = (
                "[%s] Steps %i (%i[%.1f]) @ %.1f SPS (%.1f). norm_m %.2f norm_p %.2f"
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.step,
                    self.step_per_transition(),
                    sps,
                    tot_sps,
                    total_norm_m.item(),
                    total_norm_p.item(),
                )
            )
            print_stats = [
                "total_loss_m",
                "total_loss_p",
                "img_loss",
                "fea_loss",
                "noise_loss",
                "done_loss",
                "reg_loss",
            ]
            for k in print_stats:
                if k in losses and losses[k] is not None:
                    print_str += " %s %.6f" % (
                        k,
                        losses[k].item() / self.numel_per_step,
                    )
            self._logger.info(print_str)
            self.start_time = self.timer()

            # write to log file
            stats = {
                "step": self.step,
                "real_step": self.real_step,
                "model/total_norm_m": total_norm_m.item(),
                "model/total_norm_p": total_norm_p.item(),
            }
            for k in losses.keys():
                stats["model/" + k] = (
                    losses[k].item() / self.numel_per_step
                    if k in losses and losses[k] is not None
                    else None
                )

            self.plogger.log(stats)
            if self.timing is not None:
                print(self.timing.summary())
        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10
        if self.timing is not None:
            self.timing.time("misc")

    def compute_rs_loss(self, target, rs, r_enc_logits, rv_tran, is_weights):
        k, b = self.flags.model_unroll_len, target["rewards"].shape[1]
        done_mask = target["done_mask"]
        if self.flags.model_enc_type == 0:
            rs_loss = (rs - target["rewards"]) ** 2
        else:
            target_rs_enc_v = rv_tran.encode(target["rewards"])
            rs_loss = torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(r_enc_logits, 0, 1),
                target=torch.flatten(target_rs_enc_v, 0, 1),
            )
            rs_loss = rs_loss.view(k, b)
        rs_loss = rs_loss * done_mask[:-1]
        rs_loss = torch.sum(rs_loss, dim=0)
        rs_loss = rs_loss * is_weights
        rs_loss = torch.sum(rs_loss)
        return rs_loss

    def compute_done_loss(self, target, pred_done_logits, is_weights):
        if self.flags.model_done_loss_cost > 0.0:
            done_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(
                pred_done_logits, target["dones"]
            )
            done_loss = done_loss * (~target["trun_done"]).float()[:-1]
            done_loss = torch.sum(done_loss, dim=0)
            done_loss = done_loss * is_weights
            done_loss = torch.sum(done_loss)
        else:
            done_loss = None
        return done_loss
    
    def compute_state_loss(self, tar, pred, mask, is_weights, cos=False):        
        if not cos:
            diff = tar - pred
            if not self.model_net.oned_input:                        
                state_loss = torch.mean(torch.square(diff), dim=(2, 3, 4))
            else:
                state_loss = torch.mean(torch.square(diff), dim=2)
        else:
            tar_flat = torch.flatten(tar, 2)
            pred_flat = torch.flatten(pred, 2)
            cos_sim = F.cosine_similarity(tar_flat, pred_flat, dim=2, eps=1e-08)
            state_loss = 1 - cos_sim
        state_loss = state_loss * mask
        state_loss = torch.sum(state_loss, dim=0)
        state_loss = state_loss * is_weights
        state_loss = torch.sum(state_loss)
        return state_loss

    def compute_losses_m(self, train_model_out, target, is_weights):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        initial_per_state = {sk: getattr(train_model_out, sk)[0] for sk in train_model_out._fields if sk.startswith("per")}
        if self.flags.model_mem_unroll_len > 0:
            past_env_state_norm = self.model_net.normalize(train_model_out.initial_per_state["past_real_state"])
            past_done = train_model_out.initial_per_state["past_done"]
            past_action = train_model_out.initial_per_state["past_action"]
            past_action = util.encode_action(past_action, self.model_net.action_space, one_hot=False)
            _, per_state = self.model_net.sr_net.encoder(past_env_state_norm, past_done, past_action, initial_per_state, flatten=True)

            #dbg_per_state = {sk: sv[-1] for sk, sv in train_model_out.initial_per_state.items() if sk.startswith("per")}
            #for sk in per_state.keys(): print(sk, torch.sum(torch.abs(per_state[sk] - dbg_per_state[sk])))
        else:
            per_state = initial_per_state

        env_state_norm = self.model_net.normalize(train_model_out.real_state[0])
        out = self.model_net.sr_net.forward(
            env_state_norm=env_state_norm,
            done=train_model_out.done[0],
            actions=train_model_out.action[: k + 1],
            state=per_state,
            future_env_state_norm=self.model_net.normalize(train_model_out.real_state[1:k+1]) if self.flags.noise_enable else None,
        )
        rs_loss = self.compute_rs_loss(
            target,
            out.rs,
            out.r_enc_logits,
            self.model_net.sr_net.rv_tran,
            is_weights,
        )
        done_loss = self.compute_done_loss(target, out.done_logits, is_weights)
        target_env_state_norm = self.model_net.normalize(target["env_states"])
        action = util.encode_action(train_model_out.action[1 : k + 1], self.model_net.action_space, one_hot=False)        
        if not self.flags.fea_loss_inf_bn:
            bn_stat = util.clone_bn_running_stats(self.model_net.vp_net)
        else:
            self.model_net.vp_net.train(False)
        with torch.no_grad():  
            target_xs = self.model_net.vp_net.encoder.forward_pre_mem(
                    target_env_state_norm, action, flatten=True, end_depth=self.flags.model_decoder_depth
            )
        if self.flags.model_img_loss_cost > 0.:
            img_loss = self.compute_state_loss(target_xs, out.xs, target["done_mask"][1:], is_weights, self.flags.img_fea_cos)
        else:
            img_loss = None
        if self.flags.model_fea_loss_cost > 0.:
            with torch.no_grad():                
                target_enc = self.model_net.vp_net.encoder.forward_pre_mem(
                    target_xs, action, flatten=True, depth=self.flags.model_decoder_depth
                )
            pred_enc = self.model_net.vp_net.encoder.forward_pre_mem(out.xs, action, flatten=True, depth=self.flags.model_decoder_depth)
            fea_loss = self.compute_state_loss(target_enc, pred_enc, target["done_mask"][1:], is_weights, self.flags.img_fea_cos)
        else:
            fea_loss = None        
        if not self.flags.fea_loss_inf_bn:
            util.restore_bn_running_stats(self.model_net.vp_net, bn_stat)
        else:
            self.model_net.vp_net.train(True)

        if out.noise_loss is not None:
            noise_loss = out.noise_loss
            noise_loss = noise_loss * target["done_mask"][1:]
            noise_loss = torch.sum(noise_loss, dim=0)
            noise_loss = noise_loss * is_weights
            noise_loss = torch.sum(noise_loss)
        else:
            noise_loss = None

        total_loss = self.flags.model_rs_loss_cost * rs_loss
        if self.flags.model_img_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_img_loss_cost * img_loss
        if self.flags.model_fea_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_fea_loss_cost * fea_loss
        if self.flags.model_done_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
        if self.flags.model_noise_loss_cost > 0.:
            total_loss = total_loss + self.flags.model_noise_loss_cost * noise_loss

        return {
            "rs_loss": rs_loss,
            "done_loss": done_loss,
            "img_loss": img_loss,
            "fea_loss": fea_loss,
            "noise_loss": noise_loss,
            "total_loss_m": total_loss,
        }, out.xs.detach()

    def compute_losses_p(self, train_model_out, target, is_weights, pred_xs):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        initial_per_state = {sk: getattr(train_model_out, sk)[0] for sk in train_model_out._fields if sk.startswith("per")}
        if self.flags.model_mem_unroll_len > 0:
            past_env_state_norm = self.model_net.normalize(train_model_out.initial_per_state["past_real_state"])
            past_done = train_model_out.initial_per_state["past_done"]
            past_action = train_model_out.initial_per_state["past_action"]
            past_action = util.encode_action(past_action, self.model_net.action_space, one_hot=False)
            _, per_state = self.model_net.vp_net.encoder(past_env_state_norm, past_done, past_action, initial_per_state, flatten=True)
        else:
            per_state = initial_per_state      
        
        if self.perfect_model:            
            env_state_norm = self.model_net.normalize(train_model_out.real_state)
            out = self.model_net.vp_net.forward(
                env_state_norm=env_state_norm[:k+1].view(((k+1) * b,) + env_state_norm.shape[2:]),
                x0=None,
                xs=None,
                done=train_model_out.done[:k+1].view(1, (k+1) * b,),
                actions=train_model_out.action[:k+1].view(1, (k+1) * b, -1),
                state={},
            )
            vs = out.vs.view(k+1, b)
            v_enc_logits = util.safe_view(out.v_enc_logits, (k+1, b, -1))
            policy = out.policy.view((k+1, b) + out.policy.shape[2:])
        else:
            env_state_norm = self.model_net.normalize(train_model_out.real_state[0])
            out = self.model_net.vp_net.forward(
                env_state_norm=env_state_norm,
                x0=None,
                xs=pred_xs, 
                done=train_model_out.done[0],
                actions=train_model_out.action[: k + 1],  # a_-1, ..., a_k-1                
                state=per_state,
            )
            vs = out.vs.view(k+1, b)
            if out.v_enc_logits is not None:
                v_enc_logits = util.safe_view(out.v_enc_logits, (k+1, b, -1))
            else:
                v_enc_logits = None
            policy = out.policy

        done_mask = target["done_mask"]
        if self.model_net.vp_net.predict_rd:
            rs_loss = self.compute_rs_loss(
                target,
                out.rs,
                out.r_enc_logits,
                self.model_net.vp_net.rv_tran,
                is_weights,
            )
            done_loss = self.compute_done_loss(target, out.done_logits, is_weights)

        # compute vs loss
        vs_loss = self.model_net.compute_vs_loss(
            vs=vs, 
            v_enc_logits=v_enc_logits, 
            target_vs=target["vs"],
        )
        vs_loss = vs_loss * done_mask
        vs_loss = torch.sum(vs_loss, dim=0)
        vs_loss = vs_loss * is_weights
        vs_loss = torch.sum(vs_loss)

        # compute policy loss
        if self.flags.require_prob:
            target_policy = target["action_probs"].detach()
        else:
            if self.model_net.discrete_action:
                target_policy = F.one_hot(
                    target["actions"], self.model_net.num_actions).detach().float()
            else:
                target_policy = target["actions"].detach().float()

        policy_loss = compute_cross_entropy_loss(
            policy, 
            target_policy, 
            self.model_net.discrete_action,
            self.flags.require_prob,
            is_weights, 
            mask=done_mask, 
        )

        # compute reg loss
        if self.flags.model_reg_loss_cost > 0.0:
            if self.perfect_model:
                pred_zs = out.pred_zs.view(k, b, -1)
            else:
                pred_zs = out.pred_zs.view(k + 1, b, -1)
            reg_loss = torch.mean(torch.square(pred_zs), dim=-1)
            if not self.perfect_model:
                reg_loss = reg_loss * done_mask
            reg_loss = torch.sum(reg_loss)
        else:
            reg_loss = None

        losses = {
            "vs_loss": vs_loss,
            "policy_loss": policy_loss,
            "reg_loss": reg_loss,
        }
        total_loss = (
            self.flags.model_vs_loss_cost * vs_loss
            + self.flags.model_policy_loss_cost * policy_loss
        )
        if self.model_net.vp_net.predict_rd:
            total_loss = total_loss + self.flags.model_rs_loss_cost * rs_loss
            losses["rs_loss"] = rs_loss
            if self.flags.model_done_loss_cost > 0.0:
                total_loss = total_loss + self.flags.model_done_loss_cost * done_loss
                losses["done_loss"] = done_loss
        if self.flags.model_reg_loss_cost > 0.0:
            total_loss = total_loss + self.flags.model_reg_loss_cost * reg_loss

        losses["total_loss_p"] = total_loss

        # compute priorities
        if self.flags.priority_alpha > 0.0:
            priorities = torch.absolute(vs[0] - target["vs"][0])
            priorities = priorities.detach().cpu().numpy()
        else:
            priorities = None

        return losses, priorities

    def prepare_data(self, train_model_out):
        k, b = self.flags.model_unroll_len, train_model_out.real_state.shape[1]
        ret_n = self.flags.model_return_n
        target_env_states = train_model_out.real_state
        target_rewards = train_model_out.reward[1 : k + 1]  # true reward r_1, r_2, ..., r_k
        target_action_probs = train_model_out.action_prob[1 : k + 2]  # true logits l_0, l_1, ..., l_k-1        
        target_actions = train_model_out.action[1 : k + 2]  # true actions l_0, l_1, ..., l_k-1

        if not self.flags.vp_fix_bootstrap:
            target_vs = train_model_out.baseline[ret_n + 1: ret_n + 2 + k]  # baseline ranges from v_k, v_k+1, ... v_2k
            for t in range(ret_n, 0, -1):
                target_vs = (
                    target_vs
                    * self.flags.discounting
                    * (~train_model_out.done[t : k + t + 1]).float()
                    + train_model_out.reward[t : k + t + 1]
                )
                t_done = train_model_out.truncated_done[t : k + t + 1]
                if torch.any(t_done):
                    target_vs[t_done] = train_model_out.baseline[t : k + t + 1][t_done]

        else:
            target_v = train_model_out.baseline[k + 1] # v is in the form of v_-1, v_0, .., v_k; this target_v is v_k
            target_vs = [target_v]
            for t in range(k, 0, -1):
                target_v = train_model_out.reward[t] + self.flags.discounting * target_v * (~train_model_out.done[t]).float()
                t_done = train_model_out.truncated_done[t]
                if torch.any(t_done):
                    target_v[t_done] = train_model_out.baseline[t][t_done]
                target_vs.append(target_v)
            
            target_vs.reverse()
            target_vs = torch.stack(target_vs)

        # if done on step j, r_j, v_j-1, a_j-1 has the last valid loss
        # we set all target r_j+1, v_j, a_j to 0, 0, and last a_{j+1}

        if not self.perfect_model:
            trun_done = torch.zeros(k + 1, b, dtype=torch.bool, device=self.device)
            true_done = torch.zeros(k + 1, b, dtype=torch.bool, device=self.device)
            # done_mask stores accumulated done: True, adone_1, adone_2, ..., adone_k
            for t in range(1, k + 1):
                trun_done[t] = torch.logical_or(
                    trun_done[t - 1], train_model_out.truncated_done[t]
                )
                true_done[t] = torch.logical_or(
                    true_done[t - 1], train_model_out.done[t]
                )
                if not self.flags.model_done_loss_cost > 0.0:
                    target_env_states[t, true_done[t]] = 0
                if t < k:
                    target_rewards[t, true_done[t]] = 0.0
                target_action_probs[t, true_done[t]] = target_action_probs[t - 1, true_done[t]]
                target_actions[t, true_done[t]] = target_actions[t - 1, true_done[t]]
                target_vs[t, true_done[t]] = 0.0
            if self.flags.model_done_loss_cost > 0.0:
                done_mask = (~torch.logical_or(trun_done, true_done)).float()
                target_done = torch.logical_and(~trun_done, true_done).float()[1:]
            else:
                done_mask = (~trun_done).float()
                target_done = None
        else:
            done_mask = torch.ones(k + 1, b, device=self.device)
            trun_done = None
            target_done = None

        return {
            "env_states": target_env_states[1 : k + 1],
            "rewards": target_rewards,            
            "actions": target_actions,
            "action_probs": target_action_probs,
            "vs": target_vs,
            "dones": target_done,
            "trun_done": trun_done,
            "done_mask": done_mask,
        }
    
    def gradient_step(self, loss, optimizer, scheduler, scaler=None):
        # gradient descent on loss
        if self.flags.model_optimizer == "sgd":
            loss = loss / self.numel_per_step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
                
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        optimize_params = optimizer.param_groups[0]["params"]
        if self.flags.model_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.model_grad_norm_clipping
            )
        else:
            total_norm = util.compute_grad_norm(optimize_params)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.last_epoch = (
            max(self.real_step - 1, 0)
        )  # scheduler does not support setting epoch directly
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return total_norm

    def step_per_transition(self):
        return self.step / (self.real_step - self.flags.model_warm_up_n + 1)

    def refresh_model(self):
        while True:
            weights = ray.get(
                self.param_buffer.get_data.remote("model_net")
            )  
            if weights is not None:
                self.model_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

    def refresh_actor(self):
        while True:
            weights = ray.get(
                self.actor_param_buffer.get_data.remote("actor_net")
            )  
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

    def save_checkpoint(self):
        self._logger.info("Saving model checkpoint to %s" % self.ckp_path)
        d = {
            "step": self.step,
            "real_step": self.real_step,
            "model_net_optimizer_p_state_dict": self.optimizer_p.state_dict(),
            "model_net_scheduler_p_state_dict": self.scheduler_p.state_dict(),
            "model_net_state_dict": self.model_net.state_dict(),
            "flags": vars(self.flags),
        }
        if self.flags.dual_net:
            d.update(
                {
                    "model_net_optimizer_m_state_dict": self.optimizer_m.state_dict(),
                    "model_net_scheduler_m_state_dict": self.scheduler_m.state_dict(),
                }
            )
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        if self.flags.dual_net:
            util.load_optimizer(self.optimizer_m, train_checkpoint["model_net_optimizer_m_state_dict"])
            util.load_scheduler(self.scheduler_m, train_checkpoint["model_net_scheduler_m_state_dict"])
        util.load_optimizer(self.optimizer_p, train_checkpoint["model_net_optimizer_p_state_dict"])
        util.load_scheduler(self.scheduler_p, train_checkpoint["model_net_scheduler_p_state_dict"])
        self.model_net.set_weights(train_checkpoint["model_net_state_dict"])
        self._logger.info("Loaded model checkpoint from %s" % ckp_path)

    def close(self):
        self.plogger.close()

@ray.remote
class ModelLearner(SModelLearner):
    pass
