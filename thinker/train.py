import time
import traceback
import sys
import ray
import torch
from thinker.self_play import SelfPlayWorker, PO_NET, PO_MODEL
from thinker.learn_actor import ActorLearner
from thinker.learn_model import ModelLearner
from thinker.log import LogWorker
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
import thinker.util as util

if __name__ == "__main__":
    logger = util.logger()
    logger.info("Initializing...")

    st_time = time.time()
    flags = util.parse()
    if not hasattr(flags, "cmd"):
        flags.cmd = " ".join(sys.argv)

    ray.init(
        num_cpus=int(flags.ray_cpu) if flags.ray_cpu > 0 else None,
        num_gpus=int(flags.ray_gpu) if flags.ray_gpu > 0 else None,
        object_store_memory=int(flags.ray_mem * 1024**3)
        if flags.ray_mem > 0
        else None,
    )

    num_gpus_available = torch.cuda.device_count()
    num_cpus_available = ray.cluster_resources()["CPU"]
    logger.info("Detected %d GPU %d CPU" % (num_gpus_available, num_cpus_available))

    gpu_n = min(int(num_gpus_available - 1), 3)
    if not flags.disable_auto_res:
        if flags.self_play_cpu:
            flags.gpu_num_actors = 0
            flags.gpu_num_p_actors = 0
            flags.gpu_self_play = 0
            if num_gpus_available == 1:
                flags.gpu_learn_actor = 0.5
                flags.gpu_learn_model = 0.5
            else:
                flags.gpu_learn_actor = 1
                flags.gpu_learn_model = 1
        else:
            flags.cpu_num_actors = 0  # int(num_cpus_available-8)
            flags.cpu_num_p_actors = 1
            flags.gpu_num_actors = [1, 1, 2, 2][gpu_n]
            flags.gpu_num_p_actors = [64, 32, 32, 32][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 0.5, 1][gpu_n]
            flags.gpu_learn_actor = [0.25, 0.5, 1, 1][gpu_n]
            flags.gpu_learn_model = [0.5, 1, 1, 1][gpu_n]
            if not flags.train_model:
                flags.gpu_learn_model = 0
                flags.gpu_num_actors = [2, 2, 2, 2][gpu_n]
                flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]

    if flags.merge_play_model:
        flags.cpu_num_actors = 0
        flags.gpu_num_actors = 1
        flags.gpu_num_p_actors = 64
        flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
        flags.gpu_learn_actor = [0.5, 1, 1, 1][gpu_n]
        flags.gpu_learn_model = 0
        flags.test_policy_type = -1

    if flags.merge_play_actor:
        flags.cpu_num_actors = 0
        flags.gpu_num_actors = 1
        flags.gpu_num_p_actors = flags.batch_size
        flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
        flags.gpu_learn_actor = 0
        flags.gpu_learn_model = [0.5, 1, 1, 1][gpu_n]
        flags.test_policy_type = -1

    if flags.merge_play_actor and flags.merge_play_model:
        flags.cpu_num_actors = 0
        flags.gpu_num_actors = 1
        flags.gpu_num_p_actors = flags.batch_size
        flags.gpu_self_play = 1
        flags.gpu_learn_actor = 0
        flags.gpu_learn_model = 0
        flags.test_policy_type = -1

    buffers = {
        "actor": ActorBuffer.options(num_cpus=1).remote(batch_size=flags.batch_size)
        if not flags.merge_play_actor
        else None,
        "model": ModelBuffer.options(num_cpus=1).remote(flags),
        "actor_param": GeneralBuffer.options(num_cpus=1).remote()
        if not flags.merge_play_actor
        else None,
        "model_param": GeneralBuffer.options(num_cpus=1).remote()
        if not flags.merge_play_model
        else None,
        "signal": GeneralBuffer.options(num_cpus=1).remote()
        if not flags.merge_play_model
        else None,
        "test": None,
    }

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    else:
        raise Exception("policy not supported")

    num_gpus_available = torch.cuda.device_count()
    num_gpus_self_play = (
        num_gpus_available
        - flags.gpu_learn_actor * float(flags.train_actor)
        - flags.gpu_learn_model * float(flags.train_model)
    )

    if flags.gpu_self_play > 0:
        num_self_play_gpu = max(num_gpus_self_play // flags.gpu_self_play, 0)
        logger.info(
            "Number of self-play worker with GPU: %d/%d"
            % (num_self_play_gpu, flags.gpu_num_actors)
        )
    else:
        num_self_play_gpu = -1

    logger.info(
        "Starting %d (gpu) self-play actors with %s policy"
        % (flags.gpu_num_actors, policy_str)
    )
    logger.info(
        "Starting %d (cpu) self-play actors with %s policy"
        % (flags.cpu_num_actors, policy_str)
    )

    self_play_workers = []
    if flags.gpu_num_actors > 0:
        self_play_workers.extend(
            [
                SelfPlayWorker.options(num_cpus=1, num_gpus=flags.gpu_self_play).remote(
                    buffers=buffers,
                    policy=flags.policy_type,
                    policy_params=None,
                    rank=n,
                    num_p_actors=flags.gpu_num_p_actors,
                    base_seed=flags.base_seed,
                    flags=flags,
                )
                for n in range(flags.gpu_num_actors)
            ]
        )

    if flags.cpu_num_actors > 0:
        self_play_workers.extend(
            [
                SelfPlayWorker.options(num_cpus=1, num_gpus=0).remote(
                    buffers=buffers,
                    policy=flags.policy_type,
                    policy_params=None,
                    rank=n + flags.gpu_num_actors,
                    num_p_actors=flags.cpu_num_p_actors,
                    base_seed=flags.base_seed,
                    flags=flags,
                )
                for n in range(flags.cpu_num_actors)
            ]
        )

    r_worker = [x.gen_data.remote() for x in self_play_workers]
    r_learner = []

    if flags.train_actor and not flags.merge_play_actor:
        actor_learner = ActorLearner.options(
            num_cpus=1, num_gpus=flags.gpu_learn_actor
        ).remote(buffers, 0, flags)
        r_learner.append(actor_learner.learn_data.remote())

    if flags.test_policy_type != -1:
        buffers["test"] = GeneralBuffer.remote()
        model_tester = [
            SelfPlayWorker.options(num_cpus=1).remote(
                buffers=buffers,
                policy=flags.test_policy_type,
                policy_params=None,
                rank=n + flags.gpu_num_actors + flags.cpu_num_actors,
                num_p_actors=1,
                flags=flags,
            )
            for n in range(5)
        ]
    else:
        model_tester = None

    if flags.train_model and not flags.merge_play_model:
        model_learner = ModelLearner.options(
            num_cpus=1, num_gpus=flags.gpu_learn_model
        ).remote(buffers, 0, flags, model_tester)
        r_learner.append(model_learner.learn_data.remote())

    if flags.use_wandb:
        log_worker = LogWorker.options(num_cpus=1, num_gpus=0).remote(flags)
        r_log_worker = log_worker.start.remote()

    if not flags.merge_play_actor:
        ray.get(r_learner[0])
        logger.info("Finished actor learner...")
        r_learner = r_learner[1:]
    else:
        ray.get(r_worker)
        logger.info("Finished self-play worker...")
    if not flags.merge_play_actor:
        buffers["actor"].set_finish.remote()
    buffers["model"].set_finish.remote()
    if not flags.merge_play_model:
        buffers["signal"].update_dict_item.remote("self_play_signals", "term", True)
    if len(r_learner) > 0:
        ray.get(r_learner)
    logger.info("Finished model learner...")
    if flags.use_wandb:
        log_worker.__ray_terminate__.remote()
    logger.info("Finished log worker...")
    logger.info("Time required: %fs" % (time.time() - st_time))
