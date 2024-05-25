from thinker.logger import SLogWorker
from thinker import util
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Wandb logger")
    parser.add_argument("--xpid", default="", help="Name of the run")   
    parser.add_argument("--savedir", default='../logs/__project__', help="Base log directory")       
    parser.add_argument('--project', type=str, default='', help='Name of the project.')
    parser.add_argument("--start_step", default=0, type=int, help="Step begins to be uploaded")
    log_flags = parser.parse_args() 

    if log_flags.project:
        log_flags.savedir = log_flags.savedir.replace('__project__', log_flags.project)

    flags = util.create_setting(args=[],
                                save_flags=False, 
                                ckp=True, 
                                xpid=log_flags.xpid,
                                savedir=log_flags.savedir,     
                                project=log_flags.project,                          
                                )    
    log_worker = SLogWorker(flags)
    print("Starting uploading previous entries...")
    if log_flags.start_step > 0:
        log_worker.preprocess(start_step=log_flags.start_step)
    print("Starting log worker...")
    log_worker.start()
