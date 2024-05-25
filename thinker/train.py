import time
import os
import ray
import torch
from thinker.buffer import ActorBuffer, GeneralBuffer, SelfPlayBuffer
from thinker.self_play import SelfPlayWorker
from thinker.logger import LogWorker
from thinker.main import ray_init
from thinker import util

if __name__ == "__main__":
    logger = util.logger()
    logger.info("Initializing...")

    st_time = time.time()
    flags = util.create_setting()

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
    if flags.auto_res: flags = util.alloc_res(flags, gpu_n)
    if flags.parallel_actor:
        actor_buffer = ActorBuffer.options(num_cpus=1).remote(batch_size=flags.actor_batch_size) 
        actor_param_buffer = GeneralBuffer.options(num_cpus=1).remote()  
    else:
        actor_buffer = None
        actor_param_buffer = None
    
    ray_obj_env = ray_init(flags=flags, save_flags=False, **vars(flags))
    ray_obj_env["actor_param_buffer"] = actor_param_buffer
    ray_obj_actor = {"actor_buffer": actor_buffer,
                     "actor_param_buffer": actor_param_buffer}   

    if not flags.train_actor: 
        self_play_buffer = SelfPlayBuffer.options(num_cpus=1).remote(flags=flags)
        ray_obj_actor["self_play_buffer"] = self_play_buffer

    self_play_workers = []
    self_play_workers.extend(
        [
            SelfPlayWorker.options(num_cpus=1, num_gpus=flags.gpu_self_play).remote(
                ray_obj_env=ray_obj_env,
                ray_obj_actor=ray_obj_actor,                
                rank=n,
                env_n=flags.env_n,           
                flags=flags,
            )
            for n in range(flags.self_play_n)
        ]
    )
    r_worker = [x.gen_data.remote() for x in self_play_workers]        

    if flags.use_wandb:
        log_worker = LogWorker.options(num_cpus=1, num_gpus=0).remote(flags)
        r_log_worker = log_worker.start.remote()

    return_codes = ray.get(r_worker)
    if all(return_codes):
        open(os.path.join(flags.ckpdir, 'finish'), 'a').close()
    if flags.use_wandb:
        ray.get(r_log_worker)
    logger.info("Time required: %fs" % (time.time() - st_time))