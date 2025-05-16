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

num_agents = 1
obs_size_per_agent = 16

# # Instantiate environment
env = RoboRLEnv(num_agents=num_agents)

# Train
train_fn = functools.partial(
    train, num_timesteps=2000000, num_evals=5, reward_scaling=0.1,
    episode_length=2000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=1,
    discounting=0.97, learning_rate=2e-4, entropy_cost=1e-3, num_envs=512,
    batch_size=512, seed=0, num_agents=num_agents,
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

    # plt.errorbar(
    #     x_data, y_data, yerr=y_dataerr)
    # plt.show()

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# Determine observation and action sizes
obs_size = env.observation_size * num_agents
action_size = env.action_size * num_agents

# Create the PPO networks
ppo_networks_module = ppo_networks.make_ppo_networks(
    num_agents=num_agents,
    agent_observation_size=obs_size_per_agent,
    agent_action_size=4, # Get action size from env
    policy_hidden_layer_sizes=(32,) * 4,  # Four hidden layers of size 32
    value_hidden_layer_sizes=(256,) * 5, # Five hidden layers of size 256
)

model_path = 'mjx_brax_policy_simplified'
model.save_params(model_path, params)
print("Model saved to: ", model_path)

params = model.load_params(model_path)

# inference_fn = make_inference_fn(params)
make_policy = ppo_networks.make_inference_fn(ppo_networks_module, num_agents, obs_size_per_agent)
jit_inference_fn = jax.jit(make_policy(params))

eval_env = env  # Use the instantiated environment for rollout

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(2)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 2000
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs.reshape(1, -1), act_rng)
  reshaped_ctrl = ctrl.squeeze(0)
  state = jit_step(state, reshaped_ctrl)
  rollout.append(state.pipeline_state)

  if state.done.any():
    break
  
# --- PRINT BALL POSITION AFTER THE LOOP ---
print("\n--- Ball Positions During Rollout ---")
# Get the start index for ball's x,y,z position from the environment instance
# ball_pos_start_index = eval_env._ball_qpos_start + 4 # Index of ball's x-pos
# ball_pos_end_index = eval_env._ball_qpos_start + 7   # Index after ball's z-pos

# Decide how often to print (e.g., every 'render_every' steps, or every step)
print_every = render_every # Match video frames
# print_every = 1 # Print every single step

for step_index, current_pipeline_state in enumerate(rollout):
    if step_index % print_every == 0:
        ball_qpos = current_pipeline_state.qpos # Get the qpos array
        # # Extract the ball's x, y, z coordinates
        # ball_position = ball_qpos[ball_pos_start_index:ball_pos_end_index]
        # print(f"Step {step_index}: Ball Position = {ball_position}")
# ------------------------------------------

# --- PRINT ROBOT POSITION AFTER THE LOOP ---
print("\n--- Robot 0 Positions During Rollout ---")

# Select the robot ID you want to track
robot_id_to_track = 0

# Get the qpos indices for the robot's X and Y slide joints from the environment instance
# Ensure these attributes exist and are correctly populated in your RoboRLEnv class
try:
    robot_x_qpos_index = eval_env._x_qpos_adr[robot_id_to_track]
    robot_y_qpos_index = eval_env._y_qpos_adr[robot_id_to_track]
    # You might also want the rotation (theta)
    robot_z_rot_qpos_index = eval_env._z_qpos_adr[robot_id_to_track]

    # Decide how often to print (e.g., every 'render_every' steps, or every step)
    print_every = render_every # Match video frames
    # print_every = 1 # Print every single step

    for step_index, current_pipeline_state in enumerate(rollout):
        if step_index % print_every == 0:
            qpos = current_pipeline_state.qpos # Get the qpos array

            # Extract the robot's x, y coordinates and rotation angle
            robot_x_position = qpos[robot_x_qpos_index]
            robot_y_position = qpos[robot_y_qpos_index]
            robot_z_rotation = qpos[robot_z_rot_qpos_index] # This is the angle in radians

            # You could potentially get Z position if it's not fixed, or geom positions,
            # but x/y slides usually define the planar base position.

            # print(f"Step {step_index}: Robot {robot_id_to_track} Position (x, y)=({robot_x_position:.4f}, {robot_y_position:.4f}), Rotation={robot_z_rotation:.4f} rad")

except AttributeError as e:
    print(f"Error accessing robot qpos indices in eval_env: {e}")
    print("Make sure eval_env is an instance of RoboRLEnv and _x_qpos_adr, _y_qpos_adr, _z_qpos_adr are initialized correctly.")
except IndexError as e:
     print(f"Error: robot_id_to_track ({robot_id_to_track}) might be out of bounds for the number of agents.")
# ------------------------------------------


#------------------------

desired_duration_seconds = 10.0
total_simulation_steps = int(desired_duration_seconds / env.dt)
print(f"Total simulation steps for {desired_duration_seconds} seconds: {total_simulation_steps}")
n_steps = total_simulation_steps

target_fps = 30
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
