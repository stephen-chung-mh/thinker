import time
import timeit
import os
import numpy as np
import argparse
import traceback
import ray
import torch
import torch.nn.functional as F

from thinker.core.vtrace import compute_v_trace
from thinker.core.file_writer import FileWriter
from thinker.actor_net import ActorNet
import thinker.util as util
from thinker.buffer import RetBuffer

def compute_baseline_loss(
    baseline,
    target_baseline,
    norm_stat=None,
    mask=None,
):
    target_baseline = target_baseline.detach()
    loss = (target_baseline - baseline)**2
    if mask is not None:
        loss = loss * mask
    if norm_stat is not None:
        loss = loss / norm_stat[2]
    return torch.sum(loss)

def compute_baseline_enc_loss(
    baseline_enc,
    target_baseline,
    rv_tran,
    enc_type,
    mask=None,
):
    target_baseline = target_baseline.detach()
    if enc_type == 1:
        baseline_enc = baseline_enc
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (target_baseline_enc - baseline_enc)**2
    elif enc_type in [2, 3]:
        target_baseline_enc = rv_tran.encode(target_baseline)
        loss = (
            torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(baseline_enc, 0, 1),
                target=torch.flatten(target_baseline_enc, 0, 1),
            )            
        )
        loss = loss.view(baseline_enc.shape[:2])
    if mask is not None: loss = loss * mask
    return torch.sum(loss)

def compute_pg_loss(c_action_log_prob, adv, mask=None):
    loss = -adv.detach() * c_action_log_prob
    if mask is not None: loss = loss * mask
    return torch.sum(loss)

class SActorLearner:
    def __init__(self, ray_obj, actor_param, flags, actor_net=None, device=None):
        self.flags = flags
        self.time = flags.profile
        self._logger = util.logger()

        if flags.parallel_actor:
            self.actor_buffer = ray_obj["actor_buffer"]
            self.actor_param_buffer = ray_obj["actor_param_buffer"]
            self.actor_net = ActorNet(**actor_param)
            self.refresh_actor()
            self.actor_net.train(True)                
            if self.flags.gpu_learn_actor > 0. and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:           
                self.device = torch.device("cpu")
        else:
            assert actor_net is not None, "actor_net is required for non-parallel mode"
            assert device is not None, "device is required for non-parallel mode"
            self.actor_net = actor_net
            self.device = device

        if self.device == torch.device("cuda"):
            self._logger.info("Init. actor-learning: Using CUDA.")
        else:
            self._logger.info("Init. actor-learning: Not using CUDA.")

       # initialize learning setting

        if not self.flags.actor_use_rms:
            self.optimizer = torch.optim.Adam(
                self.actor_net.parameters(), lr=flags.actor_learning_rate, eps=flags.actor_adam_eps
            )
        else:
            self.optimizer = torch.optim.RMSprop(
                self.actor_net.parameters(),
                lr=flags.actor_learning_rate,
                momentum=0,
                eps=0.01,
                alpha=0.99,
            )

        self.step = 0
        self.tot_eps = 0
        self.real_step = 0

        lr_lambda = lambda epoch: 1 - min(
            epoch * self.flags.actor_unroll_len * self.flags.actor_batch_size,
            self.flags.total_steps * self.flags.rec_t,
        ) / (self.flags.total_steps * self.flags.rec_t)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.ckp_path = os.path.join(flags.ckpdir, "ckp_actor.tar")
        if flags.ckp: self.load_checkpoint(self.ckp_path)

        # other init. variables for consume_data
        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )
        self.ret_buffers = [RetBuffer(max_actor_id, mean_n=400)]
        if self.flags.im_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=20000))
        if self.flags.cur_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=400))      
        self.im_discounting = self.flags.discounting ** (1 / self.flags.rec_t)
        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.num_rewards += int(flags.cur_cost > 0.0)
        self.norm_stats = [None,] * self.num_rewards
        self.anneal_c = 1
        self.n = 0

        # initialize file logs
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )
        
        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)        

        # variables for timing
        self.queue_n = 0
        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10

        self.disable_thinker = flags.wrapper_type == 1

    def learn_data(self):
        timing = util.Timings() if self.time else None
        try:
            while self.real_step < self.flags.total_steps:
                if timing is not None:
                    timing.reset()
                # get data remotely
           
                while True:
                    data_ptr = self.actor_buffer.read.remote()
                    data = ray.get(data_ptr)
                    if data is not None:
                        break
                    time.sleep(0.001)
                    self.queue_n += 0.001
                if timing is not None:
                    timing.time("get_data")
         
                train_actor_out, initial_actor_state = data
                train_actor_out = util.tuple_map(
                    train_actor_out, lambda x: torch.tensor(x, device=self.device)
                )
                initial_actor_state = util.tuple_map(
                    initial_actor_state, lambda x: torch.tensor(x, device=self.device)
                )
                if timing is not None:
                    timing.time("convert_data")
                data = (train_actor_out, initial_actor_state)
                # start consume data
                self.consume_data(data)
                del train_actor_out, initial_actor_state, data
                ray.internal.free(data_ptr)
                self.actor_param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
                if timing is not None:
                    timing.time("set weight")            
          
            self._logger.info("Terminating actor-learning thread")
            self.close()
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close()
            return True

    def consume_data(self, data, timing=None):
        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id

        # compute losses
        losses, train_actor_out = self.compute_losses(
            train_actor_out, initial_actor_state
        )
        total_loss = losses["total_loss"]
        if timing is not None:
            timing.time("compute loss")

        # gradient descent on loss
        self.optimizer.zero_grad()
        total_loss.backward()
        if timing is not None:
            timing.time("compute gradient")

        optimize_params = self.optimizer.param_groups[0]["params"]
        if self.flags.actor_grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.actor_grad_norm_clipping
            )
            total_norm = total_norm.detach().cpu().item()
        else:
            total_norm = util.compute_grad_norm(optimize_params)
        if timing is not None:
            timing.time("compute norm")

        self.optimizer.step()
        if timing is not None:
            timing.time("grad descent")

        self.scheduler.step()
        self.anneal_c = max(1 - self.real_step / self.flags.total_steps, 0)

        # statistic output
        stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id)
        stats["sps"] = self.sps

        # write to log file
        self.plogger.log(stats)

        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            self.sps = (
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
                "[%s] Steps %i @ %.1f SPS (%.1f). (T_q: %.2f) Eps %i. Ret %f (%f/%f). Loss %.2f"
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.sps,
                    tot_sps,
                    self.queue_n,
                    self.tot_eps,
                    stats["rmean_episode_return"],
                    stats.get("rmean_im_episode_return", 0.),
                    stats.get("rmean_cur_episode_return", 0.),
                    total_loss,
                )
            )
            print_stats = [
                "actor/entropy_loss",
                "actor/reg_loss",
                "actor/total_norm",
                "actor/mean_abs_v",
            ]
            for k in print_stats:
                print_str += " %s %.2f" % (k.replace("actor/", ""), stats[k])
            if self.flags.return_norm_type != -1:
                print_str += " norm_diff (%.4f/%.4f/%.4f)" % (
                    stats["actor/norm_diff"],
                    stats.get("actor/im_norm_diff", 0.),
                    stats.get("actor/cur_norm_diff", 0.),
                )

            self._logger.info(print_str)
            self.start_time = self.timer()
            self.queue_n = 0
            if self.time:
                print(self.timing.summary())

        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10

        if timing is not None:
            timing.time("misc")
        del train_actor_out, losses, total_loss, stats, total_norm
        torch.cuda.empty_cache()

        # update shared buffer's weights
        self.n += 1
        return self.real_step > self.flags.total_steps

    def compute_losses(self, train_actor_out, initial_actor_state):
        # compute loss and then discard the first step in train_actor_out

        T, B = train_actor_out.done.shape
        T = T - 1
        
        if self.disable_thinker:
            clamp_action = train_actor_out.pri[1:]
        else:
            clamp_action = (train_actor_out.pri[1:], train_actor_out.reset[1:])
        new_actor_out, _ = self.actor_net(
            train_actor_out, 
            initial_actor_state,
            clamp_action = clamp_action,
            compute_loss = True,
        )

        # Take final value function slice for bootstrapping.
        bootstrap_value = new_actor_out.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        train_actor_out = util.tuple_map(train_actor_out, lambda x: x[1:])
        new_actor_out = util.tuple_map(new_actor_out, lambda x: x[:-1])
        rewards = train_actor_out.reward

        # compute advantage and baseline        
        pg_losses = []
        baseline_losses = []
        discounts = [(~train_actor_out.done).float() * self.im_discounting]
        masks = [None]
        if self.flags.im_cost > 0.:
            discounts.append((~(train_actor_out.step_status == 2)).float() * self.im_discounting)            
            masks.append((~(train_actor_out.step_status == 0)).float())
        if self.flags.cur_cost > 0.:
            discounts.append((~train_actor_out.done).float() * self.im_discounting)            
            masks.append(None)
        
        log_rhos = new_actor_out.c_action_log_prob - train_actor_out.c_action_log_prob
        for i in range(self.num_rewards):
            v_trace = compute_v_trace(
                log_rhos=log_rhos,
                discounts=discounts[i],
                rewards=rewards[:, :, i],
                values=new_actor_out.baseline[:, :, i],
                values_enc=new_actor_out.baseline_enc[:, :, i]
                if new_actor_out.baseline_enc is not None
                else None,
                rv_tran=self.actor_net.rv_tran,
                enc_type=self.flags.critic_enc_type,
                bootstrap_value=bootstrap_value[:, i],
                return_norm_type=self.flags.return_norm_type,
                norm_stat=self.norm_stats[i], 
            )
            self.norm_stats[i] = v_trace.norm_stat
            pg_loss = compute_pg_loss(
                c_action_log_prob=new_actor_out.c_action_log_prob,
                adv=v_trace.pg_advantages,
                mask=masks[i]
            )
            pg_losses.append(pg_loss)
            if self.flags.critic_enc_type == 0:
                baseline_loss = compute_baseline_loss(
                    baseline=new_actor_out.baseline[:, :, i],
                    target_baseline=v_trace.vs,
                    norm_stat=self.norm_stats[i],
                    mask=masks[i]
                )
            else:
                baseline_loss = compute_baseline_enc_loss(
                    baseline_enc=new_actor_out.baseline_enc[:, :, i],
                    target_baseline=v_trace.vs,
                    rv_tran=self.actor_net.rv_tran,
                    enc_type=self.flags.critic_enc_type,
                    mask=masks[i]
                )

            baseline_losses.append(baseline_loss)
        
        # sum all the losses
        total_loss = pg_losses[0]
        total_loss += self.flags.baseline_cost * baseline_losses[0]

        losses = {
            "pg_loss": pg_losses[0],
            "baseline_loss": baseline_losses[0]
        }
        n = 0
        for prefix in ["im", "cur"]:
            cost = getattr(self.flags, "%s_cost" % prefix)
            if cost > 0.:
                n += 1
                if getattr(self.flags, "%s_cost_anneal" % prefix):
                    cost *= self.anneal_c
                total_loss += cost * pg_losses[n]
                total_loss += (cost * self.flags.baseline_cost * 
                            baseline_losses[n])
                losses["%s_pg_loss" % prefix] = pg_losses[n]
                losses["%s_baseline_loss" % prefix] = baseline_losses[n]

        # process entropy loss

        f_entropy_loss = new_actor_out.entropy_loss
        entropy_loss = f_entropy_loss * (train_actor_out.step_status == 0).float()
        entropy_loss = torch.sum(entropy_loss)        
        losses["entropy_loss"] = entropy_loss
        total_loss += self.flags.entropy_cost * entropy_loss

        if not self.disable_thinker:
            im_entropy_loss = f_entropy_loss * (train_actor_out.step_status != 0).float()
            im_entropy_loss = torch.sum(im_entropy_loss)
            total_loss += self.flags.im_entropy_cost * im_entropy_loss
            losses["im_entropy_loss"] = im_entropy_loss

        reg_loss = torch.sum(new_actor_out.reg_loss)
        losses["reg_loss"] = reg_loss
        total_loss += self.flags.reg_cost * reg_loss

        if self.flags.xss_cost > 0.:
            xss_loss = torch.sum(new_actor_out.misc["xss_loss"])
            losses["xss_loss"] = xss_loss
            total_loss += self.flags.xss_cost * xss_loss

        losses["total_loss"] = total_loss
        return losses, train_actor_out

    def compute_stat(self, train_actor_out, losses, total_norm, actor_id):
        """Update step, real_step and tot_eps; return training stat for printing"""
        stats = {}
        T, B, *_ = train_actor_out.episode_return.shape

        # extract episode_returns
        if torch.any(train_actor_out.real_done):
            episode_returns = train_actor_out.episode_return[train_actor_out.real_done][
                :, 0
            ]
            episode_returns = tuple(episode_returns.detach().cpu().numpy())
            episode_lens = train_actor_out.episode_step[train_actor_out.real_done]
            episode_lens = tuple(episode_lens.detach().cpu().numpy())
            done_ids = actor_id.broadcast_to(train_actor_out.real_done.shape)[
                train_actor_out.real_done
            ]
            done_ids = tuple(done_ids.detach().cpu().numpy())
        else:
            episode_returns, episode_lens, done_ids = (), (), ()

        self.ret_buffers[0].insert(episode_returns, done_ids)
        stats = {"rmean_episode_return": self.ret_buffers[0].get_mean()}

        n = 0
        for prefix in ["im", "cur"]:            
            if prefix == "im":
                done = train_actor_out.step_status == 2
            elif prefix == "cur":
                done == train_actor_out.real_done
            
            if getattr(self.flags, "%s_cost" % prefix) > 0.0:
                n += 1
                self.ret_buffers[n].insert_raw(
                    train_actor_out.episode_return,
                    ind=1,
                    actor_id=actor_id,
                    done=done,
                )
                r = self.ret_buffers[n].get_mean()
                stats["rmean_%s_episode_return" % prefix] = r

        if not self.disable_thinker:
            max_rollout_depth = (
                (train_actor_out.max_rollout_depth[train_actor_out.step_status==0])
                .detach()
                .cpu()
                .numpy()
            )
            max_rollout_depth = (
                np.average(max_rollout_depth) if len(max_rollout_depth) > 0 else 0.0
            )
            stats["max_rollout_depth"] = max_rollout_depth

        self.step += self.flags.actor_unroll_len * self.flags.actor_batch_size
        self.real_step += torch.sum(train_actor_out.step_status == 0).item()
        self.tot_eps += torch.sum(train_actor_out.real_done).item()
        mean_abs_v = torch.mean(torch.abs(train_actor_out.baseline)).item()

        stats.update({
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
            "actor/total_norm": total_norm,
            "actor/mean_abs_v": mean_abs_v,
        })

        if losses is not None:
            for k, v in losses.items():
                if v is not None:
                    stats["actor/"+k] = v.item()

        if self.flags.return_norm_type != -1:
            stats["actor/norm_diff"] = (
                self.norm_stats[0][1] - self.norm_stats[0][0]
                ).item()
            n = 0
            for prefix in ["im", "cur"]:
                if getattr(self.flags, "%s_cost" % prefix) > 0.:
                    n += 1
                    stats["actor/%s_norm_diff" % prefix] = (
                        self.norm_stats[n][1] - self.norm_stats[n][0]
                    ).item()
        return stats

    def save_checkpoint(self):
        self._logger.info("Saving actor checkpoint to %s" % self.ckp_path)
        d = {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "ret_buffers": self.ret_buffers,
                "actor_net_optimizer_state_dict": self.optimizer.state_dict(),
                "actor_net_scheduler_state_dict": self.scheduler.state_dict(),
                "actor_net_state_dict": self.actor_net.state_dict(),
                "flags": vars(self.flags),
            }      
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self.optimizer.load_state_dict(
            train_checkpoint["actor_net_optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(
            train_checkpoint["actor_net_scheduler_state_dict"]
        )
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])
        self._logger.info("Loaded actor checkpoint from %s" % ckp_path)

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

    def close(self):
        self.actor_buffer.set_finish.remote()
        self.plogger.close()


@ray.remote
class ActorLearner(SActorLearner):
    pass
