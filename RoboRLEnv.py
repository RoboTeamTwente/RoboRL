# Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import jax
from jax import numpy as jnp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model
from brax.training import distribution, networks

from etils import epath
from flax import linen, struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx

class RoboRLEnv(PipelineEnv):
    """Simple RoboRL environment."""

    def __init__(self, xml_path="robot_model/multi_robot_soccer_generated.xml", num_agents=1, **kwargs):

        mj_model = mujoco.MjModel.from_xml_path(xml_path) # This is on CPU
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        # Get the correct starting index for the ball's free joint in qpos and qvel
        self._ball_joint_id = mj_model.joint("ball_joint").id
        self._ball_qpos_start = mj_model.jnt_qposadr[self._ball_joint_id]
        self._ball_qvel_start = mj_model.jnt_dofadr[self._ball_joint_id]

        self._field_width = 12.0
        self._field_height = 9.0

        self._ball_geom_id = mj_model.geom("golf_ball_geom").id

        self._num_agents = num_agents
        self._robot_joints_info = {}
        self._kicker_geom_ids = {}

        self._obs_size = 16 # observation size per agent

        # _x_qpos_adr contains the starting index positions of agent n for x pos etc
        self._x_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._y_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._z_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)

        self._x_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._y_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._z_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)

        self._kicker_geom_ids = jnp.zeros(self._num_agents, dtype=jnp.int32)

        for robot_id in range(self._num_agents):
            self._x_qpos_adr = self._x_qpos_adr.at[robot_id].set(
                mj_model.jnt_qposadr[mj_model.joint(f"x_slide_{robot_id}").id]
            )
            self._y_qpos_adr = self._y_qpos_adr.at[robot_id].set(
                mj_model.jnt_qposadr[mj_model.joint(f"y_slide_{robot_id}").id]
            )
            self._z_qpos_adr = self._z_qpos_adr.at[robot_id].set(
                mj_model.jnt_qposadr[mj_model.joint(f"z_rotate_{robot_id}").id]
            )
            self._x_qvel_adr = self._x_qvel_adr.at[robot_id].set(
                mj_model.jnt_dofadr[mj_model.joint(f"x_slide_{robot_id}").id]
            )
            self._y_qvel_adr = self._y_qvel_adr.at[robot_id].set(
                mj_model.jnt_dofadr[mj_model.joint(f"y_slide_{robot_id}").id]
            )
            self._z_qvel_adr = self._z_qvel_adr.at[robot_id].set(
                mj_model.jnt_dofadr[mj_model.joint(f"z_rotate_{robot_id}").id]
            )
            self._kicker_geom_ids = self._kicker_geom_ids.at[robot_id].set(
                mj_model.geom(f"kicker_plate_geom_{robot_id}").id
            )

    def reset(self, rng: jnp.ndarray) -> State:
        """Reset the environment.
        
        Args:
            rng: A JAX random key.
            
        Returns:
            The initial state.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = jnp.zeros(self.sys.nq)
        qvel = jnp.zeros(self.sys.nv)
            
        # Ball is put on center of the field
        qpos = qpos.at[self._ball_qpos_start:self._ball_qpos_start+4].set(jnp.array([1.0, 0.0, 0.0, 0.0])) # Quartenion
        qpos = qpos.at[self._ball_qpos_start+4:self._ball_qpos_start+7].set(jnp.array([0.0, 0.0, 0.02])) # Position

        for robot_id in range(self._num_agents):
            x_pos = jax.random.uniform(rng1, minval=-(self._field_width/2), maxval=(self._field_width/2))
            y_pos = jax.random.uniform(rng2, minval=-(self._field_height/2), maxval=(self._field_height/2))

            qpos = qpos.at[self._x_qpos_adr[robot_id]].set(x_pos)
            qpos = qpos.at[self._y_qpos_adr[robot_id]].set(y_pos)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'out_of_bounds': zero,
            'left_goal': zero,
            'right_goal': zero,
            'is_dribbling' : zero,
            'is_dribbling_count': zero,
            'is_nan': zero,
        }

        return State(data, obs, reward, done, metrics)

    def _post_init(self) -> None:
        """
        """
        pass

    def _get_reward(self,
                obs: jnp.ndarray,
                is_in_left_goal: bool, 
                is_in_right_goal: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute reward function for all agents, calculating both individual rewards
        and a combined team reward.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            action: Actions taken by all agents, shape [num_agents*4].
            obs: Flattened observation array for all agents.
            is_out_of_bounds: Flag indicating if ball is out of bounds.
            is_in_left_goal: Flag indicating if ball is in left goal.
            is_in_right_goal: Flag indicating if ball is in right goal.

        Returns:
            Combined total reward for all agents.
        """

        agent_obs_batch = obs.reshape(self._num_agents, self._obs_size) # [num_agents][obs_dim]

        def _get_single_reward(agent_obs):
            """
            Compute reward for a single agent.
            
            Args:
                agent_idx: The index of the agent.
                obs: The full observation array for all agents.
                action: The action taken by this agent.
                
            Returns:
                Tuple of total reward and ball-specific reward component.
            """

            robot_pos = agent_obs[0:2]           # Robot x, y position  
            robot_vel = agent_obs[2:4]           # Robot x, y velocity
            robot_orientation = agent_obs[4:6]   # Robot orientation (cos, sin)
            ball_pos = agent_obs[6:9]            # Ball position (x, y, z)
            ball_vel = agent_obs[9:12]           # Ball velocity (x, y, z)
            is_out_of_bounds = agent_obs[12]     # Ball out of bounds flag
            is_in_left_goal = agent_obs[13]      # Ball in left goal flag
            is_in_right_goal = agent_obs[14]     # Ball in right goal flag
            is_dribbling = agent_obs[15]         # Is agent dribbling
            
            base_to_ball_vector, facing_ball_score = self._get_ball_vector_and_angle(agent_obs)
            base_to_ball_vel = jnp.dot(base_to_ball_vector, robot_vel)
            
            # Ball movement rewards
            base_to_ball_reward = jnp.where(base_to_ball_vel > 0.0, jnp.abs(base_to_ball_vel), 0.0)
            # Combine ball-related rewards
            ball_reward_component = base_to_ball_reward * 0.5 + facing_ball_score * 0.025
            
            # General rewards
            # goal_reward = jnp.where(is_in_right_goal, 10.0, 0.0)
            is_dribbling_reward = jnp.where(is_dribbling, 100.0, 0.0)
            # out_of_bounds_penalty = jnp.where(is_out_of_bounds, -1.0, 0.0)
            # action_penalty = -0.01 * jnp.sum(jnp.square(agent_action))
            
            # general_reward = goal_reward + is_dribbling_reward + out_of_bounds_penalty + action_penalty
            total_reward = is_dribbling_reward + ball_reward_component
            
            return total_reward
        
        # We vmap and parallize over all agents at once
        individual_rewards = jnp.sum(jax.vmap(_get_single_reward)(agent_obs_batch))
        team_reward = 0

        total_reward = individual_rewards + team_reward

        return total_reward

    def step(self, state: State, action: jnp.ndarray) -> State:

        """
        Runs one timestep of the environment's dynamics.

        Args:
        state: State of the environment with type mjx_env.State.
        action: Action to take with shape [num_agents*action_size_per_agent].
            Each row contains the actions for one agent.

        Returns:
        state: Updated state of the environment with type mjx_env.State.
        """
        
        # Scale the incoming actions from [-1, 1] to [-3, 3], and flatten because pipeline_step expects an array of all actuators, of lenght: num_agents * 4
        action = action * 3.0
        
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        obs = self._get_obs(data) # Huge jnp array of all agent observations

        # Extract environment conditions - these are the same for all agents
        # and are at the end of the last agent's observation
        is_out_of_bounds = obs[-4]
        is_in_left_goal = obs[-3]
        is_in_right_goal = obs[-2]
        is_dribbling = obs[-1]

        reward = self._get_reward(obs, is_in_left_goal, is_in_right_goal)

        is_nan = jnp.logical_or(
        jnp.isnan(data.qpos).any(),
        jnp.isnan(data.qvel).any()
        ).astype(jnp.float32)

        # Update metrics
        metrics = dict(state.metrics)
        metrics['out_of_bounds'] = is_out_of_bounds
        metrics['left_goal'] = is_in_left_goal
        metrics['right_goal'] = is_in_right_goal
        metrics['is_dribbling'] = is_dribbling
        metrics['is_nan'] = is_nan

        dribbling_termination = jnp.logical_and(state.done == 0, is_dribbling > 0).astype(jnp.float32)

        current_count = metrics.get('is_dribbling_count', jnp.array(0.0, dtype=jnp.float32))
        metrics['is_dribbling_count'] = current_count + dribbling_termination

        game_ending_condition = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(is_out_of_bounds, is_in_left_goal),
                jnp.logical_or(is_in_right_goal, is_nan)
            ),
            is_dribbling
        )

        # # Temporarily modify game_ending_condition to isolate the cause
        # game_ending_condition = jnp.zeros_like(is_dribbling)  # Always False

        done = game_ending_condition.astype(jnp.float32)

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics
        )

    def _get_obs(self, data: mjx.Data) -> jnp.ndarray:
        """
        Args:
            data: MuJoCo simulation data.
            info: Additional info dict (unused).
            
        Returns:
            Observation array containing robot and ball state information.
        """

        all_agent_ids = jnp.arange(self._num_agents)

        def get_single_obs(robot_id):
            robot_state = self._get_robot_state(data, robot_id)
            robot_pos = robot_state[0:2]
            robot_vel = robot_state[4:6]
            robot_orientation = robot_state[2:4]

            # Access ball position (after quaternion) and ball velocity (x, y, z)
            ball_pos = data.qpos[self._ball_qpos_start+4:self._ball_qpos_start+7]
            ball_vel = data.qvel[self._ball_qvel_start+3:self._ball_qvel_start+6]

            is_out_of_bounds = self._is_out_of_bounds(ball_pos)
            is_in_left_goal = self._is_in_left_goal(ball_pos)
            is_in_right_goal = self._is_in_right_goal(ball_pos)
            is_dribbling = robot_state[-1]

            return jnp.concatenate([
                robot_pos,                                          # (x, y)
                robot_vel,                                          # (x, y)
                robot_orientation,                                  # (cos, sin)
                ball_pos,                                           # (x, y, z)   
                ball_vel,                                           # (x, y, z)
                jnp.array([is_out_of_bounds]),                      # float32
                jnp.array([is_in_left_goal]),                       # float32
                jnp.array([is_in_right_goal]),                      # float32
                jnp.array([is_dribbling])                           # float32
            ])
        
        per_agent_obs = jax.vmap(get_single_obs)(all_agent_ids)
        flattened_obs = per_agent_obs.flatten()

        # Verify the flattened shape is num_agents * 16
        expected_length = self._num_agents * 16
        assert flattened_obs.shape[0] == expected_length, \
            f"Expected length {expected_length}, got {flattened_obs.shape[0]}"
        
        return flattened_obs
        
    def _is_out_of_bounds(self, ball_pos):
        """Check if the ball is out of bounds.

        Args:
            ball_pos: Array containing ball position [x, y, z].
            
        Returns:
            Boolean indicating whether the ball is out of bounds.
        """
        # Field dimensions from the XML (12m × 9m field)
        # The boundary lines are at x=±6.0, y=±4.5
        x, y, _ = ball_pos

        return jnp.logical_or(
            jnp.greater(jnp.abs(x), self._field_width/2),
            jnp.greater(jnp.abs(y), self._field_height/2)
        ).astype(jnp.float32)

    def _is_in_left_goal(self, ball_pos):
        """Check if the ball is in the left goal.

        Args:
            ball_pos: Array containing ball position [x, y, z].
            
        Returns:
            Boolean indicating whether the ball is in the left goal.
        """
        # Field dimensions from the XML (12m × 9m field)
        # The boundary lines are at x=±6.0, y=±4.5
        x, y, z = ball_pos

        return jnp.all(jnp.array([
            jnp.less(x, -6),            # x < -6 (inside left goal)
            jnp.less(jnp.abs(y), 0.9),  # |y| < 0.9 (within goal width)
        ])).astype(jnp.float32)
    
    def _is_in_right_goal(self, ball_pos): 

        """Check if the ball is in the right goal.

        Args:
            ball_pos: Array containing ball position [x, y, z].
            
        Returns:
            Boolean indicating whether the ball is in the right goal.
        """
        # Field dimensions from the XML (12m × 9m field)
        # The boundary lines are at x=±6.0, y=±4.5
        x, y, z = ball_pos

        return jnp.all(jnp.array([
            jnp.greater(x, 6),           # x > 6 (inside right goal)
            jnp.less(jnp.abs(y), 0.9),   # |y| < 0.9 (within goal width)
        ])).astype(jnp.float32)
    
    def _is_dribbling(self, data: mjx.Data, agent_id: int, robot_vel: jnp.ndarray) -> jnp.ndarray:
        """
        Check if the robot is dribbling with the ball at low velocity.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            robot_id: The ID of the robot (1-indexed).
            robot_vel: Velocity of the robot (x, y, z).
            
        Returns:
            Boolean indicating whether the robot is dribbling with the ball.
        """

        robot_speed = jnp.sqrt(jnp.sum(jnp.square(robot_vel))) # Squares the values in robot_vel and sums them, then take sqrt for the magnitude
        is_low_speed = robot_speed < 0.1

        number_of_contacts = data.ncon
        ball_geom_id = self._ball_geom_id
        kicker_geom_id = self._kicker_geom_ids[agent_id]
        
        # In MJX, contact data is stored in separate arrays
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        
        ball_kicker_contact_1 = jnp.logical_and(
            geom1 == ball_geom_id, 
            geom2 == kicker_geom_id
        )
        
        ball_kicker_contact_2 = jnp.logical_and(
            geom2 == ball_geom_id,
            geom1 == kicker_geom_id
        )
        
        has_ball_contact = jnp.any(jnp.logical_or(ball_kicker_contact_1, ball_kicker_contact_2)) 
        is_dribbling = jnp.logical_and(is_low_speed, has_ball_contact)

        # First get the result using where. Basically: if there are contacts, return is_dribbling, else return 0
        result = jnp.where(number_of_contacts > 0, is_dribbling, jnp.zeros((), dtype=jnp.float32))

        # Convert to jax.Array of size [1,]
        return jnp.array([result], dtype=jnp.float32)
    
    def _get_ball_vector_and_angle(self, agent_obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes both the vector from robot to ball and the angle score for how much 
        the robot is facing the ball, using pre-computed observation data.

        Args:
            agent_obs: Observation array for a single agent.
            
        Returns:
            Tuple containing:
                - Vector from the robot base to the ball (normalized).
                - Score for how much the robot is facing the ball.
        """
        robot_pos = agent_obs[0:2]           # Robot x, y position
        robot_orientation = agent_obs[4:6]   # Robot orientation (cos, sin)
        ball_pos = agent_obs[6:9]            # Ball position (x, y, z)
        
        robot_to_ball_vector = ball_pos[:2] - robot_pos  # pointing from robot to ball
        distance_L2 = jnp.sqrt(jnp.sum(jnp.square(robot_to_ball_vector))) + 1e-10
        base_to_ball_vector = robot_to_ball_vector / distance_L2

        dot_product = jnp.clip(jnp.dot(robot_orientation, base_to_ball_vector), -1.0, 1.0)
        angle = jnp.arccos(dot_product)
        facing_score = jnp.exp(-(angle/0.4)**2)

        return base_to_ball_vector, facing_score

    def _get_robot_state(self, data: mjx.Data, robot_id: int) -> jnp.ndarray:
        """
        Function to get the robot state for a specific robot using MJX.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            robot_id: The ID of the robot (1-indexed).
            
        Returns:
            Robot state as a jax.Array.
            robot_pos: Position of the robot (x, y) --> robot_state[0:2]
            robot_orientation: Orientation of the robot (cos, sin) --> robot_state[2:4]
            robot_vel: Velocity of the robot (x, y, z) --> robot_state[4:7]
        """

        x_pos = data.qpos[self._x_qpos_adr[robot_id]]
        y_pos = data.qpos[self._y_qpos_adr[robot_id]]
        rot_pos = data.qpos[self._z_qpos_adr[robot_id]]

        x_vel = data.qvel[self._x_qvel_adr[robot_id]]
        y_vel = data.qvel[self._y_qvel_adr[robot_id]]
        rot_vel = data.qvel[self._z_qvel_adr[robot_id]]

        robot_pos = jnp.array([x_pos, y_pos])
        robot_orientation = jnp.array([jnp.cos(rot_pos), jnp.sin(rot_pos)])
        robot_vel = jnp.array([x_vel, y_vel, rot_vel])

        # Check if robot is dribbling
        is_dribbling = self._is_dribbling(data, robot_id, robot_vel)
        
        # Return combined state
        return jnp.concatenate([
            robot_pos, 
            robot_orientation, 
            robot_vel, 
            is_dribbling
        ])
    
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._num_agents*4 # x_slide, y_slide, rotate, kicker_extend

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model