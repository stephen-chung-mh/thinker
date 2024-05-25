import time, timeit
import os
import re
import torch
import numpy as np
import traceback
import ray
import thinker.util as util
from thinker.actor_net import ActorNet
from thinker.main import Env

def gen_video_wandb(video_stats):
    import cv2

    # Generate video
    imgs = []
    h, w, c = video_stats["real_imgs"][0].shape

    for i in range(len(video_stats["real_imgs"])):
        img = np.zeros(shape=(h, w * 2, 3), dtype=np.uint8)
        real_img = np.copy(video_stats["real_imgs"][i])
        im_img = np.copy(video_stats["im_imgs"][i])
        if c == 1:
            real_img = np.repeat(real_img, 3, axis=2)
            im_img = np.repeat(im_img, 3, axis=2)

        if video_stats["status"][i] == 1:
            # reset; yellow tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
            im_img[:, :, 1] = 255 * 0.3 + im_img[:, :, 1] * 0.7
        elif video_stats["status"][i] == 3:
            # force reset; red tint
            im_img[:, :, 0] = 255 * 0.3 + im_img[:, :, 0] * 0.7
        elif video_stats["status"][i] == 0:
            # real reset; blue tint
            im_img[:, :, 2] = 255 * 0.3 + im_img[:, :, 2] * 0.7

        img[:, :w] = real_img
        img[:, w:] = im_img
        imgs.append(img)

    enlarge_fcator = 3
    new_imgs = []
    for img in imgs:
        new_h, new_w = h * enlarge_fcator, w * enlarge_fcator * 2
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        resized_img = np.transpose(resized_img, (2, 0, 1))
        new_imgs.append(resized_img)

    return np.array(new_imgs)


class SLogWorker:
    def __init__(self, flags):
        self.flags = flags
        self.ckpdir = flags.ckpdir
        self.actor_log_path = os.path.join(self.ckpdir, "logs.csv")
        self.model_log_path = os.path.join(self.ckpdir, "logs_model.csv")        
        self.actor_net_path = os.path.join(self.ckpdir, "ckp_actor.tar")
        self.model_net_path = os.path.join(self.ckpdir, "ckp_model.tar")        
        self.actor_fields = None
        self.model_fields = None
        self.vp_fields = None
        self.last_actor_tick = -1
        self.last_model_tick = -1
        self.last_vp_tick = -1
        self.real_step = -1
        self.last_real_step_v = -1
        self.last_real_step_c = -1
        self.vis_policy = self.flags.policy_vis_freq > 0 and not (self.flags.wrapper_type == 0 and not flags.dual_net)
        self.device = torch.device("cpu")
        self._logger = util.logger()
        self._logger.info("Initalizing log worker")
        self.log_actor = True
        self.log_model = flags.train_model
        self.log_freq = 10  # log frequency (in second)
        self.wrapper_type = self.flags.wrapper_type
        self.wlogger = util.Wandb(flags)
        self.timer = timeit.default_timer
        self.video = None
        self.env_init = False

        if self.vis_policy:            
            self.env =  Env(
                name=flags.name,
                env_n=1,   
                gpu=False,
                load_net=False,
                train_model=False,
                parallel=False,
                savedir=flags.savedir,        
                xpid=flags.xpid,
                ckp=True,                          
                base_seed=np.random.randint(10000),
                return_x=True)

            obs_space = self.env.observation_space
            action_space = self.env.action_space  
            actor_param = {
                        "obs_space":obs_space,
                        "action_space":action_space,
                        "flags":flags,
                        "tree_rep_meaning": self.env.get_tree_rep_meaning(),
                    }

            self.actor_net = ActorNet(**actor_param)
            self.actor_net.to(self.device)
            self.actor_net.train(False)
        
    @torch.no_grad()
    def start(self):
        try:
            while True:
                time.sleep(self.log_freq)

                # log stat
                self.log_stat()

                # visualize policy
                if (
                    self.vis_policy and 
                    self.real_step - self.last_real_step_v >= self.flags.policy_vis_freq
                ):
                    self._logger.info(
                        f"Steps {self.real_step}: Uploading video to wandb..."
                    )
                    self.last_real_step_v = self.real_step
                    self.visualize_wandb()
                    self._logger.info(
                        f"Steps {self.real_step}: Finish uploading video to wandb..."
                    )

                # upload files
                if (
                    self.real_step - self.last_real_step_c >= self.flags.wandb_ckp_freq
                    and self.flags.wandb_ckp_freq > 0
                ):
                    self._logger.info(
                        f"Steps {self.real_step}: Uploading files to wandb..."
                    )
                    self.last_real_step_c = self.real_step
                    self.wlogger.wandb.save(
                        os.path.join(self.ckpdir, "*")
                    )
                    self._logger.info(
                        f"Steps {self.real_step}: Finish uploading files to wandb..."
                    )

                # check if finish
                if os.path.exists(os.path.join(self.flags.ckpdir, 'finish')):
                    self.close()
                    return True

        except Exception as e:
            self._logger.error(
                f"Steps {self.real_step}: Exception detected in log_worker: {e}"
            )
            self._logger.error(traceback.format_exc())
        finally:
            self.close()
            return True

    def read_stat(self, log, fields, tick, name):
        # read the last line in log file and parse it as dict
        # if log file not yet exists or last line has not been
        # updated or fields / last line cannot be read, return None
        if fields is None:
            if os.path.exists(log):
                with open(log, "r") as f:
                    fields_ = f.readline()
                if fields_.endswith("\n"):
                    fields = fields_.strip().split(",")
                    self._logger.info(f"Steps {self.real_step}: Read fields for {name}")
                    self._logger.info(fields)
                else:
                    pass
                    # self._logger.info("Cannot read fields from %s" % log)
            else:
                self._logger.error(f"Steps {self.real_step}: File {log} does not exist")

        if fields is not None:
            stat = self.parse_line(fields, self.last_non_empty_line(log))
            if stat is not None and tick != stat["_tick"]:
                tick = stat["_tick"]
                return stat, fields, tick
        return None, fields, tick

    def log_stat(self):
        try:
            if self.log_actor:
                actor_stat, self.actor_fields, self.last_actor_tick = self.read_stat(
                    self.actor_log_path, self.actor_fields, self.last_actor_tick, "actor"
                )
            if self.log_model:
                model_stat, self.model_fields, self.last_model_tick = self.read_stat(
                    self.model_log_path,
                    self.model_fields,
                    self.last_model_tick,
                    "model",
                )
           
            stat = {}
            if self.log_model and model_stat is not None:                
                stat.update(model_stat)
                if not self.log_actor:
                    self.real_step = model_stat["real_step"]

            if self.log_actor and actor_stat is not None:
                self.real_step = actor_stat["real_step"]
                stat.update(actor_stat)

            if self.video is not None:
                stat.update(self.video)
                self.video = None
                
            excludes = ["_tick", "# _tick", "_time"]
            for y in excludes:
                stat.pop(y, None)
            if stat:
                stat["real_step"] = self.real_step
                self.wlogger.wandb.log(stat, step=self.real_step)
        except Exception as e:
            self._logger.error(
                f"Steps {self.real_step}: Error loading stat from log: {e}"
            )
            self._logger.error(traceback.format_exc())
            return None
        return

    def visualize_wandb(self):
            
        if not os.path.exists(self.actor_net_path):
            self._logger.info(
                f"Steps {self.real_step}: Actor net checkpoint {self.actor_net_path} does not exist"
            )
            return None
        if self.wrapper_type != 1:
            if not os.path.exists(self.model_net_path):
                self._logger.info(
                    f"Steps {self.real_step}: Model net checkpoint {self.model_net_path} does not exist"
                )
                return None
            try:
                checkpoint = torch.load(self.actor_net_path, torch.device("cpu"))
                self.actor_net.set_weights(checkpoint["actor_net_state_dict"])
                checkpoint = torch.load(self.model_net_path, torch.device("cpu"))
                self.env.model_net.set_weights(checkpoint["model_net_state_dict"])
            except Exception as e:
                self._logger.error(f"Steps {self.real_step}: Error loading checkpoint: {e}")
                return None

        if True: #not self.env_init:
            state = self.env.reset()
            env_out = util.init_env_out(state, self.flags, self.actor_net.dim_actions, self.actor_net.tuple_action)            
            self.actor_state = self.actor_net.initial_state(batch_size=1)
        else:
            env_out = self.last_env_out

        step = 0
        record_steps = self.flags.policy_vis_length * self.flags.rec_t
        # max_steps = record_step0s + np.random.randint(100) * self.flags.rec_t # randomly skip the first 100 real steps
        max_steps = record_steps
        start_time = self.timer()

        video_stats = {"real_imgs": [], "im_imgs": [], "status": []}
        start_record = False
        if self.flags.grayscale and "Sokoban" not in self.flags.name:
            copy_n = 1
        else:
            copy_n = 3

        while step < max_steps:
            step += 1
            actor_out, self.actor_state = self.actor_net(env_out, self.actor_state)     
            primary_action, reset_action = actor_out.action      
            state, reward, done, info = self.env.step(
                primary_action=primary_action, 
                reset_action=reset_action)
            env_out = util.create_env_out(actor_out.action, state, reward, done, info, self.flags)
            if self.wrapper_type != 1:
                ret_reset = self.env.decode_tree_reps(env_out.tree_reps)["cur_reset"]
            else:
                ret_reset = False
                
            last_step_real = (env_out.step_status == 0) | (env_out.step_status == 3)
            if last_step_real: 
                root_real_states = env_out.real_states[0, 0, -copy_n:]
                root_xs = env_out.xs[0, 0, -copy_n:]

            if start_record:                
                # record data for generating video
                if ret_reset:
                    video_stats["real_imgs"].append(root_real_states)
                    video_stats["im_imgs"].append(root_xs)
                    video_stats["status"].append(1)
                if last_step_real:
                    video_stats["real_imgs"].append(root_real_states)                    
                    video_stats["status"].append(0)
                else:
                    video_stats["real_imgs"].append(root_real_states)
                    video_stats["status"].append(2)
                video_stats["im_imgs"].append(env_out.xs[0, 0, -copy_n:])

            if (
                step >= max_steps - record_steps
                and last_step_real
                and not start_record
            ):
                video_stats["real_imgs"].append(root_real_states)
                video_stats["im_imgs"].append(env_out.xs[0, 0, -copy_n:])
                video_stats["status"].append(0) # 0 for real step, 1 for reset, 2 for normal                
                start_record = True

            if self.timer() - start_time > self.log_freq:
                start_time = self.timer()
                self.log_stat()

        self.last_env_out = env_out

        _, real_h, real_w = video_stats["real_imgs"][0].shape
        _, im_h, im_w  = video_stats["im_imgs"][0].shape
        need_resacle = real_h != im_h or real_w != im_w
        import cv2

        for l in range(len(video_stats["real_imgs"])):
            video_stats["real_imgs"][l] = np.transpose((video_stats["real_imgs"][l]).cpu().numpy(), (1, 2, 0))                    

        for l in range(len(video_stats["im_imgs"])):
            im_img = np.transpose((video_stats["im_imgs"][l]).cpu().numpy(), (1, 2, 0))
            im_img = np.clip(im_img, 0, 1) * 255
            im_img = im_img.astype(np.uint8)  
            if need_resacle:          
                im_img = cv2.resize(im_img, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
            video_stats["im_imgs"][l] = im_img

        video = gen_video_wandb(video_stats)
        self.video = {f"policy": self.wlogger.wandb.Video(video, fps=5, format="gif")}
        self.env_init = True

    def last_non_empty_line(self, file_path, delimiter="\n"):
        # A safe version of reading last line
        if os.path.getsize(file_path) <= 0:
            self._logger.error(f"Steps {self.real_step}: {file_path} is empty")
            return None
        with open(file_path, "rb") as f:
            f.seek(-1, os.SEEK_END)  # Move to the last character in the file
            last_char = f.read(1)
            # If the line does not end with '\n', it is incomplete
            if last_char != delimiter.encode():
                self._logger.error(
                    f"Steps {self.real_step}: Last line does not end with delimiter"
                )
                return None
            while f.tell() > 0:
                char = f.read(1)
                if char == delimiter.encode():
                    line = f.readline().decode().strip()
                    if line:
                        return line
                f.seek(-2, os.SEEK_CUR)
        return None

    def parse_line(self, header, line):
        if header is None or line is None:
            return None
        data = re.split(r",(?![^\(]*\))", line.strip())
        data_dict = {}
        if len(header) != len(data):
            self._logger.error(
                f"Steps {self.real_step}: Header size and data size mismatch"
            )
            print("header:", header)
            print("data:", data)
            return None
        for n, (key, value) in enumerate(zip(header, data)):
            try:
                if not value:
                    value = None
                else:
                    value = eval(value)
                if type(value) == str:
                    value = eval(value)
            except (SyntaxError, NameError, TypeError) as e:
                self._logger.error(
                    f"Steps {self.real_step}: Cannot read value {value} for key {key}: {e}"
                )
                value = None
            data_dict[key] = value
            if n == 0:
                data_dict["_tick"] = value  # assume first column is the tick
        return data_dict
    
    # functions for uploading past entry only
    def preprocess(self, start_step):
        stats = self.read_file(self.actor_log_path, start_step)
        if self.flags.train_model:
            stats_m = self.read_file(self.model_log_path, start_step)
            stats = self.merge_stat(stats, stats_m)
        print("Uploading data with size %d for %s..." % (len(stats), self.flags.xpid))
        print(f"Example of data uploaded {stats[-1]}")

        for n, stat in enumerate(stats):        
            self.wlogger.wandb.log(stat, step=stat['real_step'])
            time.sleep(0.1)
            if n % 50 == 0: print("Uploading step: %d" % stat['real_step'])

    def merge_stat(self, stats_a, stats_b):
        l = min(len(stats_a), len(stats_b))
        stats = []
        for n in range(l):
            stat = stats_b[n]
            stat.update(stats_a[n])
            stats.append(stat)
        return stats

    def read_file(self, file, start_step, freq=5000):    
        stats = []
        with open(file, 'r') as f:
            fields_ = f.readline()
            fields = fields_.strip().split(',')
            cur_real_step = 0
            while(True):
                line = f.readline()
                if not line: break
                out = self.parse_line(fields, line)   
                if out['real_step'] > cur_real_step:
                    if out['real_step'] > start_step: stats.append(out)
                    cur_real_step += freq
        return stats

    def close(self):
        self.wlogger.wandb.finish()
        return


@ray.remote
class LogWorker(SLogWorker):
    pass


if __name__ == "__main__":
    flags = util.parse(override=True)
    log_worker = SLogWorker(flags)
    log_worker.start()