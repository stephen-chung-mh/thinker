import torch
import time
import numpy as np
import argparse
import ray
import os
from thinker.self_play import SelfPlayWorker, PO_NET, PO_MODEL, PO_NSTEP
from thinker.buffer import GeneralBuffer
import thinker.util as util
import datetime


def save_np_array_to_csv(array, path, prefix, logger):
    test_folder = os.path.join(path, "test")
    os.makedirs(test_folder, exist_ok=True)
    # Generate the current timestamp for the file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{prefix}test_{timestamp}.csv"
    file_path = os.path.join(test_folder, file_name)
    column_vector = array.reshape(-1, 1)
    np.savetxt(file_path, column_vector, delimiter=",")
    logger.info(f"Saved results to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test for Thinker v{util.__version__}"
    )
    parser.add_argument(
        "--load_checkpoint", required=True, help="Load checkpoint directory."
    )
    parser.add_argument(
        "--policy_type",
        default=0,
        type=int,
        help="Policy used for self-play worker; 0 for actor net, 1 for model policy, 2 for 1-step greedy",
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Whether to select greedy action."
    )
    parser.add_argument(
        "--test_eps_n", default=100, type=int, help="Number of episode to test for"
    )
    parser.add_argument(
        "--ray_mem",
        default=32,
        type=float,
        help="Memory allocated to ray object store in GB.",
    )
    parser.add_argument(
        "--ray_cpu",
        default=-1,
        type=int,
        help="Manually allocate number of cpu for ray.",
    )
    parser.add_argument(
        "--ray_gpu",
        default=-1,
        type=int,
        help="Manually allocate number of gpu for ray.",
    )
    parser.add_argument(
        "--env", type=str, default="", help="Gym environment (override)."
    )
    parser.add_argument(
        "--base_seed", default=1, type=int, help="Base seed of environment."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full test: test all three variants at the same time; required 2 GPUs.",
    )
    parser.add_argument(
        "--test_rec_t",
        default=-1,
        type=int,
        help="Whether to use a different K in inference.",
    )
    flags = parser.parse_args()
    assert flags.test_eps_n % 100 == 0, f"test_eps_n must be multiple of 100"

    arg_list = []
    for arg_name, arg_value in vars(flags).items():
        if arg_name not in ["test_eps_n", "greedy", "base_seed", "full", "rec_t"]:
            arg_list.extend([f"--{arg_name}", str(arg_value)])
    flags_ = util.parse(arg_list)
    for key, value in vars(flags_).items():
        if flags.env and key == "env":
            continue
        setattr(flags, key, value)

    logger = util.logger()
    logger.info(f"Testing {flags.xpid} on {flags.env} for {flags.test_eps_n} episodes")
    ray.init(
        num_cpus=int(flags.ray_cpu) if flags.ray_cpu > 0 else None,
        num_gpus=int(flags.ray_gpu) if flags.ray_gpu > 0 else None,
        object_store_memory=int(flags.ray_mem * 1024**3)
        if flags.ray_mem > 0
        else None,
    )

    st_time = time.time()

    param_buffer = GeneralBuffer.remote()
    test_buffer = GeneralBuffer.remote()
    buffers = {
        "actor": None,
        "model": None,
        "actor_param": GeneralBuffer.options(num_cpus=1).remote(),
        "model_param": GeneralBuffer.options(num_cpus=1).remote(),
        "signal": GeneralBuffer.options(num_cpus=1).remote(),
        "test": test_buffer,
    }

    if flags.policy_type == PO_NET:
        policy_str = "actor network"
    elif flags.policy_type == PO_MODEL:
        policy_str = "base model network"
    elif flags.policy_type == PO_NSTEP:
        policy_str = "n-step greedy search"

    flags.num_actors = 2
    flags.num_p_actors = 50
    flags.train_actor = False
    flags.train_model = False
    flags.preload_actor = os.path.join(flags.load_checkpoint, "ckp_actor.tar")
    flags.preload_model = os.path.join(flags.load_checkpoint, "ckp_model.tar")

    num_gpus_available = torch.cuda.device_count()
    if not flags.full:
        num_gpus = 1 if num_gpus_available >= 2 else 0.5
    else:
        num_gpus = 1 if num_gpus_available >= 4 else 0.5

    if flags.full:
        flags.greedy = False
        flags.policy_type = 0

    test_eps_n_per_actor = flags.test_eps_n // (flags.num_actors * flags.num_p_actors)

    if flags.greedy:
        policy_params = {"greedy": True}
    else:
        policy_params = None

    logger.info("Starting %d actors with %s policy" % (flags.num_actors, policy_str))
    self_play_workers = [
        SelfPlayWorker.options(num_cpus=0, num_gpus=num_gpus).remote(
            buffers=buffers,
            policy=flags.policy_type,
            policy_params=policy_params,
            rank=n,
            num_p_actors=flags.num_p_actors,
            flags=flags,
            base_seed=flags.base_seed,
        )
        for n in range(flags.num_actors)
    ]

    r_worker = [
        x.gen_data.remote(test_eps_n=test_eps_n_per_actor) for x in self_play_workers
    ]

    if flags.full:
        logger.info("Starting %d actors with greedy policy" % (flags.num_actors))
        greedy_test_buffer = GeneralBuffer.options(num_cpus=1).remote()
        buffers["test"] = greedy_test_buffer
        self_play_workers = [
            SelfPlayWorker.options(num_cpus=0, num_gpus=num_gpus).remote(
                buffers=buffers,
                policy=0,
                policy_params={"greedy": True},
                rank=n + flags.num_actors,
                num_p_actors=flags.num_p_actors,
                flags=flags,
                base_seed=flags.base_seed,
            )
            for n in range(flags.num_actors)
        ]
        r_worker.extend(
            [
                x.gen_data.remote(test_eps_n=test_eps_n_per_actor)
                for x in self_play_workers
            ]
        )

    ray.get(r_worker)

    all_returns = ray.get(test_buffer.get_data.remote("episode_returns"))
    all_returns = np.array(all_returns)
    prefix = ""
    if flags.greedy and flags.policy_type == 0:
        prefix += "greedy_"
    if flags.policy_type == 1:
        prefix += "model_"
    if flags.test_rec_t > 0:
        prefix += "%dK_" % flags.test_rec_t
    if flags.env == "cSokoban-test-v0":
        prefix += "test_"

    save_np_array_to_csv(
        array=all_returns, path=flags.load_checkpoint, prefix=prefix, logger=logger
    )

    if flags.full:
        all_returns = ray.get(greedy_test_buffer.get_data.remote("episode_returns"))
        all_returns = np.array(all_returns)
        save_np_array_to_csv(
            array=all_returns,
            path=flags.load_checkpoint,
            prefix="greedy_",
            logger=logger,
        )

    logger.info("Time required: %fs" % (time.time() - st_time))
