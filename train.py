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

import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['MUJOCO_GL'] = 'osmesa'
jax.config.update("jax_default_matmul_precision", "high")  

from RoboRLEnv import RoboRLEnv

num_agents = 1
obs_size_per_agent = 16

# # Instantiate environment
env = RoboRLEnv(num_agents=num_agents)

relative_ckpt_path = epath.Path('./models_3')
ckpt_path = relative_ckpt_path.resolve()
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f'{current_step}'
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)

# Train
train_fn = functools.partial(
    train, num_timesteps=30000000, num_evals=15, reward_scaling=1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=40, num_minibatches=20, num_updates_per_batch=8,
    discounting=0.999, learning_rate=1e-4, entropy_cost=3e-3, num_envs=1024, 
    batch_size=1024, seed=0, num_agents=num_agents,
    obs_size_per_agent=16, policy_params_fn=policy_params_fn, restore_checkpoint_path=ckpt_path / '68812800')

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

    # plt.errorbar(
    #     x_data, y_data, yerr=y_dataerr)
    # plt.show()

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# Determine observation and action sizes
obs_size = env.observation_size * num_agents
action_size = env.action_size * num_agents

# Save model
model_path = 'mjx_brax_policy_simplified_v3_part2'
model.save_params(model_path, params)
print("Model saved to: ", model_path)

# Load Model and Define Inference Function
params = model.load_params(model_path)

# Visualize Policy
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

eval_env = env
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 1000
render_every = 2

print_once = True 
for i in range(n_steps):
  if print_once:
    print("--- INFERENCE/VISUALIZATION ---")
    raw_obs_inference = state.obs
    print(f"Raw observation from RoboRLEnv (state.obs): {raw_obs_inference}")
    print(f"Raw obs shape: {raw_obs_inference.shape}, dtype: {raw_obs_inference.dtype}")

    # This is what your jit_inference_fn receives after reshape
    obs_for_inference_fn = state.obs.reshape(1, -1)
    print(f"Observation fed to jit_inference_fn (after reshape): {obs_for_inference_fn}")
    print(f"Input obs shape: {obs_for_inference_fn.shape}, dtype: {obs_for_inference_fn.dtype}")
    print("---------------------------")
    print_once = False

  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs.reshape(1, -1), act_rng)
  reshaped_ctrl = ctrl.squeeze(0)
  state = jit_step(state, reshaped_ctrl)
  rollout.append(state.pipeline_state)

  if state.done:
    break

# Decide how often to print (e.g., every 'render_every' steps, or every step)
print_every = render_every # Match video frames
desired_duration_seconds = 10.0
total_simulation_steps = int(desired_duration_seconds / env.dt)
print(f"Total simulation steps for {desired_duration_seconds} seconds: {total_simulation_steps}")
n_steps = total_simulation_steps

target_fps = 31.2
render_every = int(1.0 / (env.dt * target_fps))
if render_every < 1:
    render_every = 1
fps = 1.0 / env.dt / render_every
print(f"Target FPS: {target_fps}, calculated render_every: {render_every}, actual FPS: {fps}")

# Define your desired resolution
render_width = 1280
render_height = 720 # Example: 720p HD

# Save the video to MP4 using the specified width and height
filename = "loaded_model_rollout_large.mp4" # Changed filename slightly
media.write_video(
    filename,
    eval_env.render(
        rollout[::render_every],
        camera='top',
        width=render_width,  # Pass the desired width
        height=render_height # Pass the desired height
    ),
    fps=fps
)
print(f"Rollout saved to: {filename} with resolution {render_width}x{render_height}.")


# export MUJOCO_GL=osmesa
# python /home/usergpu/RoboRL/train.py