__version__ = "1.1.0"
__project__ = "thinker"

import collections
import time
import timeit
import yaml
import argparse
import subprocess
import os
import logging
from matplotlib import pyplot as plt
import numpy as np
import torch
import re
import sys

def process_flags(flags):

    if flags.wrapper_type == 1:
        flags.rec_t = 1
        flags.train_model = False
        flags.im_enable = False
        flags.cur_enable = False
        flags.return_h = False
        flags.return_double = False

    if flags.wrapper_type == 2:
        flags.dual_net = False
        flags.cur_enable = False
        flags.model_rs_loss_cost = 0
        flags.model_img_loss_cost = 0
        flags.model_done_loss_cost = 0

    return flags

def process_flags_actor(flags):    
    if flags.wrapper_type == 1:
        flags.see_h = False
        flags.see_x = False
        flags.see_tree_rep = False
    flags.return_h = flags.see_h
    flags.return_x = flags.see_x
    if flags.wrapper_type != 0:
        flags.im_cost = 0.
        flags.cur_cost = 0.
    return flags

def alloc_res(flags, gpu_n):
    if flags.auto_res:
        flags.self_play_n = [1, 1, 2, 2][gpu_n]
        flags.env_n = [64, 32, 32, 32][gpu_n]
        flags.gpu_self_play = [0.25, 0.5, 0.5, 1][gpu_n]
        flags.gpu_learn_actor = [0.25, 0.5, 1, 1][gpu_n]
        flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.train_model:
            flags.gpu_learn = 0
            flags.self_play_n = [2, 2, 2, 2][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.train_actor:
            flags.gpu_learn_actor = 0
            flags.self_play_n = [2, 2, 2, 3][gpu_n]
            flags.gpu_self_play = [0.25, 0.5, 1, 1][gpu_n]
        if not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = 64
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn = 0
        if not flags.parallel_actor:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = [0.5, 1, 1, 1][gpu_n]
            flags.gpu_learn_actor = 0
            flags.gpu_learn = [0.5, 1, 1, 1][gpu_n]
        if not flags.parallel_actor and not flags.parallel:
            flags.self_play_n = 1
            flags.env_n = flags.actor_batch_size
            flags.gpu_self_play = 1
            flags.gpu_learn_actor = 0
            flags.gpu_learn = 0
    return flags

def add_parse(filename, parser=None, prefix=''):
    # Load default configuration
    if type(filename) is not list: 
        filename = [filename]
    config = {}
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # Set up command line argument parsing
    if parser is None:
        parser = argparse.ArgumentParser(description=f"{__project__} v{__version__}")
    try:
        parser.add_argument('--config', type=str, help="Path to user's thinker configuration file")
    except:
        # if there is dulplicate key, just ignore
        pass

    if prefix and prefix[-1] != "_": prefix = prefix + "_"
    # Dynamically add command line arguments based on the default config keys and their types
    for key, value in config.items():
        try:
            if isinstance(value, bool):
                parser.add_argument(f'--{prefix}{key}', type=lambda x: (str(x).lower() == 'true'), help=f"Override {key}")
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), help=f"Override {key}")
        except:
            # if there is dulplicate key, just ignore
            pass
    return parser

def create_flags(filename, save_flags=True, post_fn=None, **kwargs):
    """create flags, a namespace object that contains the config; the load
       order is filename[0], filename[1], ..., kwargs['config'], kwargs       
       args:
            filename (str/list of str): the config file(s) to load
            save_flags (bool): weather to save the flags
            post_fn (function): a function that takes flags and output flags
            **kwargs: all other settings 
       return:
            flags (namespace): config                
    """
    if type(filename) is not list: 
        filename = [filename]

    config = {}
    for n in filename:
        default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', n)
        with open(default_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # If user provided their own YAML configuration, load it and update defaults
    if "config" in kwargs and kwargs["config"]:
        with open(kwargs["config"], 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    # Check for command line argument overrides and apply them
    for key in config.keys():
        if key in kwargs and kwargs[key] is not None:
            config[key] = kwargs[key]            

    # Convert dictionary to named tuple
    flags = argparse.Namespace(**config)    

    # additional info
    flags.savedir = flags.savedir.replace("__project__", __project__)    
    flags.__version__ = __version__
    flags.cmd = " ".join(sys.argv) 

    try:
        flags.git_revision = get_git_revision_hash()
    except Exception:
        flags.git_revision = None

    if flags.ckp:
        # load setting from checkpoint yaml    
        xpid = 'latest' if not flags.xpid else flags.xpid
        config_path = os.path.join(flags.savedir, xpid, "config_c.yaml")        
        if os.path.islink(config_path): config_path = os.readlink(config_path)
        with open(config_path, 'r') as f:
            config_ = yaml.safe_load(f)
        for key, value in config_.items():
            if (key not in ['ckp', 'ray_mem', 'ray_gpu', 'savedir'] and
                not (key in kwargs and kwargs[key] is not None)):
                setattr(flags, key, value)
        print("Loaded config from %s" % config_path)

    if not flags.xpid:        
        flags.xpid = "%s-%s" % (__project__, time.strftime("%Y%m%d-%H%M%S"))

    flags.ckpdir = os.path.join(flags.savedir, flags.xpid,)     

    flags = process_flags(flags)
    if post_fn is not None: flags = post_fn(flags)

    if save_flags and not flags.ckp:        
        ckpdir = full_path(flags.ckpdir)
        if not os.path.exists(ckpdir):   
            os.makedirs(ckpdir)        
        try:
            # create sym link for the latest run
            symlink = os.path.join(full_path(flags.savedir), "latest")
            if os.path.islink(symlink):
                os.remove(symlink)
            if not os.path.exists(symlink):
                os.symlink(flags.ckpdir, symlink)
                print("Symlinked log directory: %s" % symlink)
        except OSError:
            # os.remove() or os.symlink() raced. Don't do anything.
            pass

        config_path = os.path.join(full_path(flags.savedir), 
                                   flags.xpid, 
                                   "config_c.yaml")
        with open(config_path, 'w') as outfile:
            yaml.dump(vars(flags), outfile)
        print("Wrote config file to %s" % config_path)  

    fs = ["savedir", "preload", "ckpdir"]
    for f in fs:
        path = getattr(flags, f)
        if path:            
            setattr(flags, f, full_path(path))
    return flags

def create_setting(args=None, save_flags=True, **kwargs):
    filenames = ['default_thinker.yaml', 'default_actor.yaml']
    parser = add_parse(filenames)
    if args is not None:
        parse_flags = parser.parse_args(args)
    else:
        parse_flags = parser.parse_args()

    parse_dict = vars(parse_flags)
    for key in parse_dict.keys():
        if key in kwargs and kwargs[key] is not None:
            parse_dict[key] = kwargs[key]            

    flags = create_flags(filenames, 
                         save_flags=save_flags, 
                         post_fn=process_flags_actor, 
                         **parse_dict)
    return flags

def full_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(path):
        path = os.readlink(path)
    return path

def tuple_map(x, f):
    def process_element(y):
        # Apply function to dictionary items
        if isinstance(y, dict):
            return {k: f(v) if v is not None else None for k, v in y.items()}
        return f(y) if y is not None else None

    if type(x) == tuple:
        return tuple(process_element(y) for y in x)
    else:
        return type(x)(*(process_element(y) for y in x))

def dict_map(x, f):
    return {k:f(v) for (k, v) in x.items()}

def safe_view(x, dims):
    if x is None:
        return None
    else:
        return x.view(*dims)


def safe_squeeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.squeeze(dim)


def safe_unsqueeze(x, dim=0):
    if x is None:
        return None
    else:
        return x.unsqueeze(dim)


def safe_concat(xs, attr, dim=0):
    if len(xs) == 0:
        return None
    if getattr(xs[0], attr) is None:
        return None
    return torch.concat([getattr(i, attr).unsqueeze(dim) for i in xs], dim=dim)


def construct_tuple(x, **kwargs):
    return x(**{k: kwargs[k] if k in kwargs else None for k in x._fields})


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

def enc(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + (0.001) * x

def dec(x):
    return np.sign(x) * (
        np.square(
            (np.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001)
        )
        - 1
    )

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def logger():
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logs/out")
    if not logger.hasHandlers():
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        logger.addHandler(shandle)
    logger.setLevel(logging.INFO)
    return logger

class Timings:
    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self._mean_deques = {}
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

        if name not in self._mean_deques:
            self._mean_deques[name] = collections.deque(maxlen=5)
        self._mean_deques[name].append(x)

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v**0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        mean_deques = self._mean_deques
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fms (last 5: %.6fms) +- %.6fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * np.average(mean_deques[k]),
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fms" % (1000 * total)
        return result

class Wandb:
    def __init__(self, flags, subname=""):
        import wandb

        self.wandb = wandb
        exp_name = flags.xpid + subname
        tags = []
        if subname == "_model":
            tags.append("model")
        m = re.match(r"^v\d+", exp_name)
        if m:
            tags.append(m[0])
        self.wandb.init(
            project=__project__,
            config=flags,
            entity=os.getenv("WANDB_USER", ""),
            reinit=True,
            # Restore parameters
            resume="allow",
            id=exp_name,
            name=exp_name,
            tags=tags,
        )
        self.wandb.config.update(flags, allow_val_change=True)

def compute_grad_norm(parameters, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
        norm_type,
    )
    return total_norm

def decode_tree_reps(tree_reps, num_actions, enc_type=0):
    idx1 = num_actions * 5 + 5
    idx2 = num_actions * 10 + 7
    d = dec if enc_type != 0 else lambda x: x
    if len(tree_reps.shape) == 3:
        tree_reps = tree_reps[0]
    return {
        "root_action": tree_reps[:, :num_actions], # action at root node
        "root_r": d(tree_reps[:, [num_actions]]), # reward at root node (should be zero)
        "root_v": d(tree_reps[:, [num_actions + 1]]), # value at root node
        "root_logits": tree_reps[:, num_actions + 2 : 2 * num_actions + 2], # policy logit at root node
        "root_qs_mean": d(tree_reps[:, 2 * num_actions + 2 : 3 * num_actions + 2]), # child mean rollout return at root node
        "root_qs_max": d(tree_reps[:, 3 * num_actions + 2 : 4 * num_actions + 2]), # child max rollout return at root node
        "root_ns": tree_reps[:, 4 * num_actions + 2 : 5 * num_actions + 2], # visit count at root node
        "root_trail_r": d(tree_reps[:, [5 * num_actions + 2]]), # trailing sum of reward till the current node
        "root_trail_q": d(tree_reps[:, [5 * num_actions + 3]]), # trailing rollout return till the current node
        "root_max_v": d(tree_reps[:, [5 * num_actions + 4]]), # maximum roolout return at the root node
        "cur_action": tree_reps[:, idx1 : idx1 + num_actions], # all the below are the same as above, except applied to the current node
        "cur_r": d(tree_reps[:, [idx1 + num_actions]]),
        "cur_v": d(tree_reps[:, [idx1 + num_actions + 1]]),
        "cur_logits": tree_reps[
            :, idx1 + num_actions + 2 : idx1 + 2 * num_actions + 2
        ],
        "cur_qs_mean": d(
            tree_reps[:, idx1 + 2 * num_actions + 2 : idx1 + 3 * num_actions + 2]
        ),
        "cur_qs_max": d(
            tree_reps[:, idx1 + 3 * num_actions + 2 : idx1 + 4 * num_actions + 2]
        ),
        "cur_ns": tree_reps[
            :, idx1 + 4 * num_actions + 2 : idx1 + 5 * num_actions + 2
        ],
        "reset": tree_reps[:, idx2], # whether reset is triggered
        "time": tree_reps[:, idx2 + 1 : -1], # step within current stage
        "derec": tree_reps[:, [-1]], # accumulated discount
        "raw": tree_reps, # the raw tree representation
    }

def mask_tree_rep(tree_reps, num_actions):
    idx1 = num_actions * 5 + 5
    idx2 = num_actions * 10 + 7
    N, C = tree_reps.shape
    rec_t = (C - 1) - (idx2 + 1)
    tree_reps_m = torch.zeros(N, 2*num_actions+3+rec_t, device=tree_reps.device)
    tree_reps_m[:, :num_actions] = tree_reps[:, :num_actions] # root_action
    tree_reps_m[:, num_actions:2*num_actions] = tree_reps[:, idx1 : idx1 + num_actions] # cur_action
    tree_reps_m[:, 2*num_actions] = tree_reps[:, idx2] # reset_action
    tree_reps_m[:, 2*num_actions+1] = tree_reps[:, idx1 + num_actions] # imagainary reward
    tree_reps_m[:, 2*num_actions+2] = tree_reps[:, -1] # deprec
    tree_reps_m[:, 2*num_actions+3:] = tree_reps[:, idx2 + 1 : -1] # time    
    return tree_reps_m

def plot_raw_state(x, ax=None, title=None, savepath=None):
    if ax is None:
        _, ax = plt.subplots()
    x = torch.swapaxes(torch.swapaxes(x[-3:].detach().cpu(), 0, 2), 0, 1).numpy()
    if x.dtype == float:
        x = np.clip(x, 0, 1)
    if x.dtype == int:
        x = np.clip(x, 0, 255)
    ax.imshow(x, interpolation="nearest", aspect="auto")
    if title is not None:
        ax.set_title(title)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, title + ".png"))
        plt.close()