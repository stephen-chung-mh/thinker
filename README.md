
# Thinker: Learning to Plan and Act

This is the official repository for the paper titled [*Thinker: Learning to Plan and Act*](https://arxiv.org/abs/2307.14993). Please refer to the [project website](https://stephen-c.com/thinker) for details of the algorithm.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Available Environments](#available-environments)
- [Resource Management](#resource-management)
- [Resuming from a Checkpoint](#resuming-from-a-checkpoint)
- [API](#api)
	- [Thinker.make](#thinkermake-function)
	- [env.reset](#envreset-method)
	- [env.step](#envstep-method)
	- [env.close](#envclose-method)
- [IMPALA on Thinker-augmented MDP](#impala-on-thinker-augmented-mdp)
- [Miscellaneous](#miscellaneous)


## Prerequisites

Ensure that Pytorch is installed (versions v2.0.0 and v1.13.0 have been tested).

## Installation

1. Update essential packages and install Cython:

```bash
sudo apt-get update
sudo apt-get install zip python-opencv build-essential -y
pip install Cython
```

2. Install the C++ version of Sokoban (skip this step if you're not running experiments on Sokoban):
```bash
cd sokoban
pip install -e .
```

3. Compile and install Thinker:
```bash
cd thinker
pip install -e .
```

## Basic Usage

The Thinker-augmented MDP provides the same interface as OpanAI's Gym. To test if the installation is successful, run the following:

```py
import thinker
import numpy as np
env_n = 16 # batch size
env = thinker.make("Sokoban-v0", env_n=env_n, gpu=False) # or atari games like "BreakoutNoFrameskip-v4"
initial_state = env.reset()
for _ in range(20):
	primary_action = np.random.randint(5, size=env_n) # 5 possible actions in Sokoban
	reset_action = np.random.randint(2, size=env_n) # 2 possible reset actions
	state, reward, done, info = env.step(primary_action, reset_action) 
print(state["tree_reps"].shape, state["real_states"].shape)
env.close()
```
which should output `torch.Size([16, 79]) torch.Size([16, 3, 80, 80])`.

## Configuration

The default configuration of the Thinker-augmented MDP can be found in `thinker/thinker/config/default_thinker.yaml`, which contains a list of parameters. Important parameters include:
* `rec_t`: the stage length K, which is `20` by default
* `max_depth`: the maximum search depth L, which is `5` by default
* `model_unroll_len`: model unroll length when training the model, which is `5` by default
* `wrapper_type`: the type of the augmented MDP; `0`: default Thinker; `1`: raw env; `2`: Thinker with a perfect state-reward network (i.e. using the true environment dynamic instead).  Using type `2` can significantly increase learning speed, at the expense of more calls to the underlying real environment. Detault: `0`
* `model_size_nn`: integer multipler to the model size. Increase it for a larger model. Default: `1`
* `xpid`: experiment id, which is used for checkpoint resumption. Default: `thinker-{DATE}-{TIME}`

Please refer to `default_thinker.yaml` for other parameters.

There are two methods to change the default parameters. The first method is to pass the parameters when calling `Thinker.make`, such as:

```py
env = thinker.make("Sokoban-v0", env_n=16, rec_t=10, max_depth=10)
```
The second method is to create a custom configuration file which contains the parameters that need to be changed, such as:

```
# custom.yaml
rec_t: 10
max_depth: 10
```
and pass it to `Thinker.make`:
```py
env = Thinker.make("Sokoban-v0", env_n=16, config='custom.yaml')
```
Note that the first method will take precedence over the second method.

## Available Environments

By default, environments like `Sokoban-v0`, as well as other Atari environments such as `BreakoutNoFrameskip-v4` and `SeaquestNoFrameskip-v4`, can be passed to the `Thinker.make` method. The `Thinker.make` method internally calls `gym.make` using the provided environment name to instantiate the environment. For all environments except `Sokoban-v0`, a pre-defined Atari wrapper is applied. This Atari wrapper includes: 
- 4-frame stacking, 
- up to 30 random no-ops at the start, 
- resizing the image to 3x84x84, 
- truncating at 108,000 steps, 
- treating end-of-life as end-of-episode.

You can also provide custom environments. To do this, supply an `env_fn` function that, when invoked as `env_fn()`, returns a Gym environment:

```py
thinker.make(env_fn=env_fn, env_n=16)
``` 

The only requirement is that the `observation_space` of the returned environment must be a `Box` with dimensions `C, H, W`. The `low` and `high` properties should be properly set (they will be used to normalize the input during model training). Additionally, the `action_space` should be of type `Discrete`. Thinker will not apply any additional wrappers.

**Notes**:

1.  If frame stacking is used in a custom environment, set the property `frame_stack_n` of the returned environment from `env_fn()` to the counts of stacking. This will ensure that the model only predicts the most recent state and not the previously stacked states. If frame stacking is detected, the log will output: `Detected frame stacking with {n} counts`.
2.  The C++ version of the Sokoban gym environment can be used as follows:

```py
import gym, gym_sokoban
env = gym.make("Sokoban-v0")
```


## Resource Management

By default, a GPU will be utilized in the Thinker-augmented MDP. To disable GPU usage, set `gpu=False` when calling `thinker.make`. The Thinker-augmented MDP offers three operational modes.

### Mode 1: Single environment with shared model training (Default)

This is the default mode. In this mode, model training occurs within the `env.step` method. Only a single environment class is permitted. Each `env` instance returned by `thinker.make` utilizes a distinct model, so it is advisable not to invoke `thinker.make` multiple times.

### Mode 2: Single environment with parallel model training

This mode employs Ray to establish a parallel thread that trains the model, as opposed to training within the `env.step` method. The `env.step` method will periodically synchronize with the model training thread to ensure the models are updated. To activate this mode, set `parallel=True` and assign `gpu_learn` to the GPU fraction allocated for the model training thread. For example,

```py
env = thinker.make("Sokoban-v0", env_n=16, parallel=True, gpu_learn=0.5)
```

### Mode 3: Multiple environments with parallel model training

This mode resembles Mode 2 but permits multiple environment classes that share a common model. To implement this, first generate a shared Ray resource. This resource can then be provided to other Ray actors to initialize new environments:

```py
ray_obj = ray_init(gpu_learn=0.5, rec_t=10)  # pass the configurations here, rather than in thinker.make
# in ray actor 1
env_1 = thinker.make("Sokoban-v0", env_n=16, ray_obj=ray_obj)  # only name, env_fn, env_n, and gpu can be passed here
# in ray actor 2
env_2 = thinker.make("Sokoban-v0", env_n=16, ray_obj=ray_obj)
```
See `thinker/train.py` and `thinker/thinker/self_play.py` for an example of using mode 3 to implement IMPALA on the Thinker-augmented MDP.

**Note**:

1.  In Modes 2 and 3, the memory, GPU, and CPU allocations for Ray can be configured using the `ray_mem`, `ray_GPU`, and `ray_CPU` parameters, respectively. If Ray returns an error, consider setting these parameters manually.


## Resuming from a Checkpoint

To resume from a previously saved checkpoint:
1.  Identify the `XPID` (Experiment ID) of the run you want to resume. By default, it is in the format: `Thinker-{DATE}-{TIME}`. This ID should be displayed in the standard output during your experiment run.    
2.  Set the `ckp` parameter to `True` when invoking `thinker.make`. For those using mode 3, you should set these parameters in the `ray_init` function.
    

Example:

```py
env = thinker.make("Sokoban-v0", env_n=16, xpid=XPID, ckp=True)
```

This command will attempt to load the checkpoint from the `savedir/XPID` directory.

## API

### `thinker.make` function

The `thinker.make` method creates a Gym-class environment. 

**Definition:**
```py
env = thinker.make(name=None, env_fn=None, ray_obj=None, env_n=1, gpu=True, **kwargs)
```
**Parameters:**

-   `name`: Name of the environment, which will be passed internally to `Gym.make`.
-   `env_fn`: Custom environment function that, when called, returns a Gym-class environment. Either `name` or `env_fn` must be provided.
-   `ray_obj`: Ray shared resources, only required for mode 3 (see Resource Management above).
-   `env_n`: Integer representing the batch size of the environment.
-   `gpu`: Boolean indicating whether a GPU will be used.
-   `**kwargs`: All other configurations present in `thinker/thinker/config/default_thinker.yaml`. Additionally, accepts `config` that points to a custom `yaml` file. For mode 3, pass `**kwargs` to `thinker.ray_init` instead of here.

**Returns:**

-   `env`: A Gym-class environment that supports the methods `reset`, `step`, and `close`.

### `env.reset` method

The `env.reset` method resets the current environment, and shall be called only once after `thinker.make`. Note that environment will be automatically reset upon episode termination and one does not need to call this.

**Definition:**
```py
initial_state = env.reset()
```
**Returns:**
-   `initial_state`: The initial state, which is in the same format as the `state` returned in `env.step` method.


### `env.step` method

The `env.step` method advances the environment's state by one time step based on the provided primary action and reset action. The function returns the next state, reward, termination indicator, and additional information.

**Definition:**
```py
state, reward, done, info = env.step(primary_action, reset_action, action_prob=None)
```
**Parameters:**

-   `primary_action`: Imaginary/real action to be taken in the environment.    
    -   Type: Torch tensor, numpy array, or list.
    -   Shape: `(env_n,)`
    
-   `reset_action`: Reset action to be taken in the environment state.    
    -   Type: Torch tensor, numpy array, or list.
    -   Shape: `(env_n,)`
    -   Each element must be either 0 or 1.
    
-   `action_prob` (Optional): Probability distribution over actions. This is required when `require_prob=True` is set for the environment. Passing the action probability provides a better training target for the model.  If `require_prob=False`, this `action_prob` will not be used.
    -   Type: Torch tensor or numpy array.
    -   Shape: `(env_n, num_actions)`
    -   Default: `None`

**Returns:**

-   `state`: A dictionary containing:
    -   `tree_reps`: Tree representation of the state.
        -   Shape: `(env_n, N)`
    -   `real_states`: The real state of the environment, which remains unchanged throughout a stage.
        -   Shape: `(env_n, C, H, W)`, where `(C, H, W)` represents the shape of the real observation space.
    -   `hs` (optional): Model's hidden state of the current node.
        -   Shape: `(env_n, hC, hH, hW)`, where `(hC, hH, hW)` represents the shape of the model's hidden state.
    -   `xs` (optional): Predicted state of the current node.
        -   Shape: `(env_n, C, H, W)`, where `(C, H, W)` represents the shape of the real observation space.
-   `reward`: Float tensor representing the reward obtained after executing the action. This is clipped by `reward_clip`  if `reward_clip` is enabled (default disabled).
	-   Shape: `(env_n,)`
-   `done`: Boolean tensor that indicates whether the episode has concluded for each environment instance.    
	-   Shape: `(env_n,)`
-   `info`: A dictionary containing:    
    -   `real_done`: Boolean tensor indicating the genuine end of the episode. It may not match `done` in Atari games, which is set to `True` whenever a life is lost.
	-   `truncated_done`: Boolean tensor indicating whether the termination occurred due to truncation.
	-   `step_status`: Integer tensor representing the current step's status. Values can be:
	    -   `0`: A real action was just taken, marking the beginning of a stage.
	    -   `1`: An imaginary action was just taken, and the next action is an imaginary action.
	    -   `2`: An imaginary action was just taken, and the next action is a real action. Typically, a stage has a `step_status` of `[0, 1, 1, ..., 1, 2]` , where the count of `1`s equals the stage length `K` minus 2.
	-   `max_rollout_depth`: Integer tensor representing the greatest depth achieved in all rollouts during the current stage; primarily used for logging.    
	-   `baseline`: Float tensor representing the mean rollout return at the root node. It is updated only at the end of a stage and can be utilized as a value estimation of the real state underlying the stage.    
	-   `im_reward`: Float tensor representing the planning rewards.
	-   `episode_step`: Integer tensor representing the episode steps within the augmented MDP.    
	-   `episode_return`: Float tensor representing the undiscounted sum of unclipped reward in the current episode (as defined by `real_done`); primarily used for performance evaluation.      
	-   `im_episode_return`: Float tensor representing the undiscounted sum of planning rewards in the current stage.    
	-   `model_status`: A dictionary providing the model's training status, including:
	    -   `processed_n`: Integer representing the number of real transitions processed by the replay buffer. It should approximately match the count of real transitions performed.
	    -   `model_warm_up_n`: Integer representing the number of real transitions processed by the replay buffer before initiating the model's training.
	    -   `running`: Boolean that denotes if the model is under training. By default, the model only begins its training after `processed_n >= model_warm_up_n`.
	    -   `finish`: Boolean indicating if the model has completed its training. The model ceases training once `processed_n >= total_steps`.

Unless stated otherwise, all output elements in `info` are Torch tensors of the data type `torch.float32` with shape `(env_n,)`.

**Note**
1.  The specific shape of each element in `state` can be checked with `env.observation_space`.
2.  `hs` in `state` is only returned when `return_h=True` (default value is `True`) and `xs` in `state` is returned only when `return_x=True` (default value is `False`).
3.  If `return_double=True` (default value is `False`), then both `hs` and `xs` will also return their respective statistics at the root node. These statistics will be stacked on top of the respective statistics at the current node in the `C` dimension.
4. To interpret the information in `tree_reps`, one can use `thinker.util.decode_tree_reps`:
```py
thinker.util.decode_tree_reps(state['tree_reps'], num_actions)
```
where `num_actions` denotes the number of action available in the real environment. This will return a dictionary that maps each elements in the tree representation to interpretable keys such as `root_logits`, `root_qs_mean`, `root_ns` etc. (see the function `decode_tree_reps` for explaination of each key).

### `env.close` method

The `env.close` method closes the environment, freeing the memory of the model.


**Definition:**
```py
env.close()
```

## IMPALA on Thinker-augmented MDP
Run the following commands in the `thinker` directory:

Sokoban default run:

```bash
python train.py
```

The above run is used to generate the results of the Thinker-augmented MDP in Figure 5 of the paper.

Atari default run (change the environment if needed):

```bash
python train.py --env BreakoutNoFrameskip-v4 --reward_clip 1 --model_size_nn 2 --discounting 0.99
```


The above run, combined with that from other Atari games, are used to generate the results of the Thinker-augmented MDP in Figure 9 of the paper.


### Visualizing a Run

To visualize a specific run of your experiment, follow the steps below:

1.  Identify the `XPID` (Experiment ID) of the run you want to visualize. By default, it is in the format: `Thinker-{DATE}-{TIME}`. This ID should be displayed in the standard output during your experiment run.    
2.  Use the following command, replacing `XPID` with the appropriate experiment ID:
    
```bash
python visual.py --xpid XPID
```

**Note**

1. If you encounter errors related to Ray memory, try setting `--ray_mem -1` to allow Ray to allocate memory automatically.
2. The number of GPUs will be detected automatically. A single RTX3090 is sufficient for the Sokoban default run, while two RTX3090s are required for the Atari default run. The number of CPUs and GPUs allocated can be controlled with the `--ray_cpu` and `--ray_gpu` options, e.g., `--ray_cpu 16 --ray_gpu 1` limits usage to 16 CPUs and 1 GPU.

## Miscellaneous

### Re-compile
If you are editing the Cython files, run the following in the `sokoban` or `thinker` folder to recompile the code:
```bash
python setup.py build_ext
```
### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Contact
For any questions or discussions, please contact me at mhc48@cam.ac.uk.