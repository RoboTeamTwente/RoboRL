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
# from brax.training.agents.ppo import networks as ppo_networks
from customPPO import customNetworksPPO as ppo_networks
from customPPO.customTrainPPO import train
# from brax.training.agents.ppo import train
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

num_agents = 2

# Instantiate environment
env = RoboRLEnv(num_agents=num_agents)

# Train
train_fn = functools.partial(
    train, num_timesteps=5000, num_evals=5, reward_scaling=0.1,
    episode_length=1000, normalize_observations=False, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=1,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=512,
    batch_size=128, seed=0, num_agents=num_agents,
    obs_size_per_agent=16,)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

max_y, min_y = 150, 0
def progress(num_steps, metrics):

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    y_dataerr.append(metrics['eval/episode_reward_std'])

    # Print all available metrics
    print("\nMetrics at step {}:".format(num_steps))
    for key, value in metrics.items():
        print(f"{key}: {value}")

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=y_dataerr)
    plt.show()

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

model_path = 'mjx_brax_policy'
model.save_params(model_path, params)
print("Model saved to: ", model_path)

params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# eval_env = envs.get_environment(env)

# jit_reset = jax.jit(eval_env.reset)
# jit_step = jax.jit(eval_env.step)

# # initialize the state
# rng = jax.random.PRNGKey(0)
# state = jit_reset(rng)
# rollout = [state.pipeline_state]

# # grab a trajectory
# n_steps = 500
# render_every = 2

# for i in range(n_steps):
#   act_rng, rng = jax.random.split(rng)
#   ctrl, _ = jit_inference_fn(state.obs, act_rng)
#   state = jit_step(state, ctrl)
#   rollout.append(state.pipeline_state)

#   if state.done:
#     break

# media.show_video(env.render(rollout[::render_every], camera='side'), fps=1.0 / env.dt / render_every)