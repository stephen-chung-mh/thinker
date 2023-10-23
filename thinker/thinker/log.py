import time, timeit
import os
import re
import torch
import numpy as np
import traceback
import ray
import thinker.util as util
from thinker.net import ActorNet, ModelNet
from thinker.env import Environment


def gen_video_wandb(video_stats, grayscale):
    import cv2

    # Generate video
    imgs = []
    hw = video_stats["real_imgs"][0].shape[1]
    copy_n = 1 if grayscale else 3

    for i in range(len(video_stats["real_imgs"])):
        img = np.zeros(shape=(copy_n, hw, hw * 2), dtype=np.uint8)
        real_img = np.copy(video_stats["real_imgs"][i])
        im_img = np.copy(video_stats["im_imgs"][i])

        if video_stats["status"][i] == 1:
            if grayscale:
                im_img[0, :, :] = im_img[0, :, :] * 0.7
            else:
                im_img[0, :, :] = 255 * 0.3 + im_img[0, :, :] * 0.7
                im_img[1, :, :] = 255 * 0.3 + im_img[1, :, :] * 0.7
        elif video_stats["status"][i] == 0:
            if grayscale:
                im_img[0, :, :] = 255 * 0.7 + im_img[0, :, :] * 0.3
            else:
                im_img[2, :, :] = 255 * 0.3 + im_img[2, :, :] * 0.7

        img[:, :, :hw] = real_img
        img[:, :, hw:] = im_img
        imgs.append(img)

    enlarge_fcator = 3
    new_imgs = []
    for img in imgs:
        _, height, width = img.shape
        new_height, new_width = height * enlarge_fcator, width * enlarge_fcator
        img = np.transpose(img, (1, 2, 0))
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        if not grayscale:
            resized_img = np.transpose(resized_img, (2, 0, 1))
        new_imgs.append(resized_img)

    return np.array(new_imgs)


class SLogWorker:
    def __init__(self, flags):
        self.flags = flags
        if flags.load_checkpoint:
            self.check_point_path = flags.load_checkpoint
        else:
            self.check_point_path = "%s/%s" % (flags.savedir, flags.xpid)
        self.actor_log_path = os.path.join(self.check_point_path, "logs.csv")
        self.model_log_path = os.path.join(self.check_point_path, "logs_model.csv")
        self.actor_net_path = os.path.join(self.check_point_path, "ckp_actor.tar")
        self.model_net_path = os.path.join(self.check_point_path, "ckp_model.tar")
        self.actor_fields = None
        self.model_fields = None
        self.last_actor_tick = -1
        self.last_model_tick = -1
        self.real_step = -1
        self.last_real_step_v = -1
        self.last_real_step_c = -1
        self.vis_policy = self.flags.policy_vis_freq > 0
        self.device = torch.device("cpu")
        self._logger = util.logger()
        self._logger.info("Initalizing log worker")
        self.log_model = flags.train_model and not self.flags.disable_model
        self.log_freq = 10  # log frequency (in second)
        self.wlogger = util.Wandb(flags)
        self.timer = timeit.default_timer
        self.video = None
        self.grayscale = flags.grayscale
        self.env_init = False

        if self.vis_policy:
            self.env = Environment(
                flags, model_wrap=True, debug=True, device=self.device
            )
            self.env.seed(np.random.randint(10000))

            self.actor_net = ActorNet(
                obs_shape=self.env.model_out_shape,
                gym_obs_shape=self.env.gym_env_out_shape,
                num_actions=self.env.num_actions,
                flags=flags,
            )
            self.actor_net.to(self.device)
            self.actor_net.train(False)
            self.model_net = ModelNet(
                obs_shape=self.env.gym_env_out_shape,
                num_actions=self.env.num_actions,
                flags=flags,
                debug=True,
            )
            self.model_net.train(False)

    @torch.no_grad()
    def start(self):
        try:
            while True:
                time.sleep(self.log_freq)

                # log stat
                self.log_stat()

                # visualize policy
                if (
                    self.real_step - self.last_real_step_v >= self.flags.policy_vis_freq
                    and self.flags.policy_vis_freq > 0
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
                        os.path.join(self.check_point_path, "*"), self.check_point_path
                    )
                    self._logger.info(
                        f"Steps {self.real_step}: Finish uploading files to wandb..."
                    )

        except Exception as e:
            self._logger.error(
                f"Steps {self.real_step}: Exception detected in log_worker: {e}"
            )
            self._logger.error(traceback.format_exc())
        finally:
            self.close(0)
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

            if actor_stat is not None:
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
        if not os.path.exists(self.model_net_path):
            self._logger.info(
                f"Steps {self.real_step}: Model net checkpoint {self.model_net_path} does not exist"
            )
            return None
        try:
            checkpoint = torch.load(self.actor_net_path, torch.device("cpu"))
            self.actor_net.set_weights(checkpoint["actor_net_state_dict"])
            checkpoint = torch.load(self.model_net_path, torch.device("cpu"))
            self.model_net.set_weights(checkpoint["model_net_state_dict"])
        except Exception as e:
            self._logger.error(f"Steps {self.real_step}: Error loading checkpoint: {e}")
            return None

        if not self.env_init:
            env_out = self.env.initial(self.model_net)
            self.actor_state = self.actor_net.initial_state(
                batch_size=1, device=self.device
            )
        else:
            env_out = self.last_env_out

        step = 0
        record_steps = self.flags.policy_vis_length * self.flags.rec_t
        # max_steps = record_step0s + np.random.randint(100) * self.flags.rec_t # randomly skip the first 100 real steps
        max_steps = record_steps
        start_time = self.timer()

        video_stats = {"real_imgs": [], "im_imgs": [], "status": []}
        start_record = False

        copy_n = 1 if self.grayscale else 3

        while step < max_steps:
            step += 1
            actor_out, self.actor_state = self.actor_net(env_out, self.actor_state)
            action = [actor_out.action, actor_out.im_action, actor_out.reset_action]
            action = torch.cat(action, dim=-1).unsqueeze(0)
            env_out = self.env.step(action, self.model_net)

            if start_record:
                if self.flags.perfect_model:
                    gym_env_out = env_out.gym_env_out
                else:
                    gym_env_out = (torch.clamp(self.env.env.xs, 0, 1) * 255).to(
                        torch.uint8
                    )
                # record data for generating video
                if action[0, 0, 2] == 1:
                    video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
                    video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
                    video_stats["status"].append(1)
                if env_out.cur_t[0, 0] == 0:
                    video_stats["real_imgs"].append(
                        gym_env_out[0, 0, -copy_n:].cpu().numpy()
                    )
                    video_stats["status"].append(0)
                else:
                    video_stats["real_imgs"].append(video_stats["real_imgs"][-1])
                    video_stats["status"].append(2)
                video_stats["im_imgs"].append(gym_env_out[0, 0, -copy_n:].cpu().numpy())

            if (
                step >= max_steps - record_steps
                and env_out.cur_t.item() == 0
                and not start_record
            ):
                video_stats["real_imgs"].append(
                    env_out.gym_env_out[0, 0, -copy_n:].cpu().numpy()
                )
                video_stats["im_imgs"].append(video_stats["real_imgs"][-1])
                video_stats["status"].append(
                    0
                )  # 0 for real step, 1 for reset, 2 for normal
                start_record = True

            if self.timer() - start_time > self.log_freq:
                start_time = self.timer()
                self.log_stat()

        self.last_env_out = env_out
        video = gen_video_wandb(video_stats, self.grayscale)
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
                return None
            data_dict[key] = value
            if n == 0:
                data_dict["_tick"] = value  # assume first column is the tick
        return data_dict

    def close(self, exit_code):
        self.wlogger.wandb.finish(exit_code=exit_code)


@ray.remote
class LogWorker(SLogWorker):
    pass


if __name__ == "__main__":
    flags = util.parse(override=True)
    log_worker = SLogWorker(flags)
    log_worker.start()
