

from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
import xml

xml_path = "robot.xml"

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

x_motor_id = 0      # First actuator is x_motor
y_motor_id = 1      # Second actuator is y_motor
rotate_motor_id = 2  # Third actuator is rotate_motor

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
    # Apply sinusoidal control to create movement
    time = mj_data.time
    
    # Apply control directly using the known indices
    mj_data.ctrl[x_motor_id] = np.sin(time) * 5  # Move in X
    mj_data.ctrl[y_motor_id] = np.cos(time) * 5  # Move in Y
    mj_data.ctrl[rotate_motor_id] = np.sin(time * 2)  # Rotate
    
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)