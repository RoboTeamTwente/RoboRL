from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

import mediapy as media
import matplotlib.pyplot as plt

import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

from RoboRLEnv import RoboRLEnv

# Instantiate environment
env = RoboRLEnv()

from mujoco_playground.config import dm_control_suite_params

# Clone the existing config
ppo_params = dict(dm_control_suite_params.brax_ppo_config('CartpoleBalance'))

# Modify specific parameters
ppo_params['num_timesteps'] = 100000  # Change total training steps
ppo_params['learning_rate'] = 1e-4  # Adjust learning rate
ppo_params['num_envs'] = 512  # Change number of parallel environments
ppo_params['batch_size'] = 16  # Make it 32 or 64 for fast learning
ppo_params['episode_length'] = 2000 # Approx 20 seconds

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    clear_output(wait=True)
    
    current_time = datetime.now()
    times.append(current_time)
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    
    # Print raw reward for debugging
    print(f"Episode reward: {metrics['eval/episode_reward']:.6f}")
    print(f"Detailed metrics: {metrics}")

    if num_steps == 0:
      print(f"Time taken to initialize: {(current_time - times[0]).total_seconds():.1f} seconds")
    
    # Calculate training speed
    if len(times) > 1 and num_steps > 0:
        steps_since_last = num_steps - x_data[-2] if len(x_data) > 1 else num_steps
        time_since_last = (current_time - times[-2]).total_seconds()
        steps_per_second = steps_since_last / time_since_last if time_since_last > 0 else 0
        
        # Only calculate estimated time if we have a valid steps_per_second
        if steps_per_second > 0:
            remaining_steps = ppo_params["num_timesteps"] - num_steps
            remaining_time = remaining_steps / steps_per_second
            print(f"Estimated time remaining: {remaining_time:.1f} seconds")
    
    # Rest of your plotting code
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    
    plt.show()

ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress
)

from mujoco_playground import wrapper

make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

