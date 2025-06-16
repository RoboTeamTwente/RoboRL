#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
import time

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

from RoboRLEnv import RoboRLEnv

"""
Run this script in the beginning to validate your .xml file.
"""

# Load model and print some info statements
mj_model = mujoco.MjModel.from_xml_path("robot.xml")
print("Model Details:")
print("Number of bodies:", mj_model.nbody)
print("Number of joints:", mj_model.njnt)
print("Number of geoms:", mj_model.ngeom)

# Init env
env = RoboRLEnv()

print("\nDetailed Initialization Timing:")
start_total = time.time()

# Time XML loading
start = time.time()
mj_model = mujoco.MjModel.from_xml_path(env._xml_path)
print(f"XML Loading time: {time.time() - start:.4f} seconds")

# Time MjData creation
start = time.time()
mj_data = mujoco.MjData(mj_model)
print(f"MjData creation time: {time.time() - start:.4f} seconds")

# Time MJX model conversion
start = time.time()
mjx_model = mjx.put_model(mj_model)
print(f"MJX model conversion time: {time.time() - start:.4f} seconds")

# Time MJX data conversion
start = time.time()
mjx_data = mjx.put_data(mj_model, mj_data)
print(f"MJX data conversion time: {time.time() - start:.4f} seconds")

# Time mjx_env.init
start = time.time()
data = mjx_env.init(mjx_model)
print(f"mjx_env.init time: {time.time() - start:.4f} seconds")

print(f"\nTotal initialization time: {time.time() - start_total:.4f} seconds")
