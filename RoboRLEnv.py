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

    def __init__(self, xml_path="robot_model/multi_robot_soccer_generated_simplified.xml", num_agents=1, **kwargs):

        mj_model = mujoco.MjModel.from_xml_path(xml_path) # This is on CPU
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 16 # approx 60hz, was 8. Now 16 at approx 30hz
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        # Get the correct starting index for the ball's free joint in qpos and qvel
        self._ball_x_slide_qpos_adr = mj_model.jnt_qposadr[mj_model.joint("ball_x_slide").id]
        self._ball_y_slide_qpos_adr = mj_model.jnt_qposadr[mj_model.joint("ball_y_slide").id]
        self._ball_x_slide_qvel_adr = mj_model.jnt_dofadr[mj_model.joint("ball_x_slide").id]
        self._ball_y_slide_qvel_adr = mj_model.jnt_dofadr[mj_model.joint("ball_y_slide").id]
        
        self._field_width = 12.0
        self._field_height = 9.0

        self._ball_geom_id = mj_model.geom("golf_ball_geom").id
        self.ball_geom_z_pos = 0.0115

        self._num_agents = num_agents
        self._robot_joints_info = {}

        self._obs_size = 16 # observation size per agent

        # _x_qpos_adr contains the starting index positions of agent n for x pos etc
        self._x_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._y_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._z_qpos_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)

        self._x_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._y_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._z_qvel_adr = jnp.zeros(self._num_agents, dtype=jnp.int32)

        self._kicker_geom_ids = jnp.zeros(self._num_agents, dtype=jnp.int32)
        self._kicker_beam_rf_ids = jnp.full(self._num_agents, -1, dtype=jnp.int32)

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

            try:
                sensor_name = f"kicker_beam_rf_{robot_id}" 
                sensor_id = mj_model.sensor(sensor_name).id
                self._kicker_beam_rf_ids = self._kicker_beam_rf_ids.at[robot_id].set(sensor_id)
                print(f"DEBUG: Agent {robot_id} Kicker Rangefinder Sensor ID = {sensor_id} (Name: {sensor_name})")
            except KeyError: 
                print(f"ERROR: Sensor '{sensor_name}' not found in the MJCF model for agent {robot_id}.")
            
        # print(f"DEBUG: Ball Geom ID = {self._ball_geom_id} (Name: {mj_model.geom(self._ball_geom_id).name})")
        # print(f"DEBUG: Kicker Geom ID[0] = {self._kicker_geom_ids[0]}")
        # print(f"DEBUG: Ball X slide qpos adr: {self._ball_x_slide_qpos_adr}")
        # print(f"DEBUG: Ball Y slide qpos adr: {self._ball_y_slide_qpos_adr}")

        self.far_distance_threshold = 0.25

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
        
        #####################
        ##### IMPORTANT #####
        #####################
        
        # The qpos and qvel we extract are with respect to the origin of the bodies.
        # So if the bodies are spawned at anything other than 0,0, the coordinate systems will be off.
        # So pay attention to the XML file; spawn everything at (0,0), or adjust for it here.

        #####################
        ##### IMPORTANT #####
        #####################

        # Define a small margin to avoid spawning exactly on the edge
        # pos_margin = 2
        # x_min = (-self._field_width / 2) + pos_margin
        # x_max = (self._field_width / 2) - pos_margin
        # y_min = (-self._field_height / 2) + pos_margin
        # y_max = (self._field_height / 2) - pos_margin

        x_min = -5
        x_max = 0
        y_min = -3.5
        y_max = 3.5

        # Split RNG key for ball and agents
        rng, ball_x_rng = jax.random.split(rng)
        rng, ball_y_rng = jax.random.split(rng)

        # # Generate random positions for the ball
        ball_x = jax.random.uniform(ball_x_rng, minval=1, maxval=5)
        ball_y = jax.random.uniform(ball_y_rng, minval=-3.5, maxval=3.5)

        # Generate static positions for the ball
        # ball_x = 2
        # ball_y = 0

        # Ball is put on center of the field
        qpos = qpos.at[self._ball_x_slide_qpos_adr].set(ball_x)
        qpos = qpos.at[self._ball_y_slide_qpos_adr].set(ball_y)
        # jax.debug.print("Generated ball position: x={x}, y={y}", x=ball_x, y=ball_y)

        initial_robot_positions = []

        for robot_id in range(self._num_agents):
            # Split rng for this agent's x, y, and rotation
            rng, agent_x_rng = jax.random.split(rng)
            rng, agent_y_rng = jax.random.split(rng)
            rng, agent_rot_rng = jax.random.split(rng)

            x_pos = jax.random.uniform(agent_x_rng, minval=x_min, maxval=x_max)
            y_pos = jax.random.uniform(agent_y_rng, minval=y_min, maxval=y_max)
            rot_pos = jax.random.uniform(agent_rot_rng, minval=-jnp.pi, maxval=jnp.pi)
            # x_pos = 1.75
            # y_pos = 0
            # rot_pos = 0

            initial_robot_positions.append(jnp.array([x_pos, y_pos]))

            qpos = qpos.at[self._x_qpos_adr[robot_id]].set(x_pos)
            qpos = qpos.at[self._y_qpos_adr[robot_id]].set(y_pos)
            qpos = qpos.at[self._z_qpos_adr[robot_id]].set(rot_pos)
        
        initial_dribble_pos = jnp.zeros((self._num_agents, 2))

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'out_of_bounds': zero,
            'left_goal': zero,
            'right_goal': zero,
            'is_dribbling' : zero,
            'is_nan': zero,
            'did_dribble_this_episode': zero,       # This will become the 0-1 trigger
            '_internal_dribble_latch': zero,        # New: internal latch
            'attempted_kick_while_dribbling': zero,

            # Rewards
            'rew_goal': zero,
            'rew_ball_to_goal': zero,
            'rew_base_to_ball': zero,
            'rew_ball_facing': zero,
            'rew_dribbling': zero,
            'pen_out_of_bounds': zero,

            # Dribbling/ tracking penalties
            'dribbling_distance_from_start': zero,
            'pen_dribbling': zero,
        }

        info = {
            'steps': 0,
            'truncation': jnp.zeros(()),
            'initial_dribble_pos': jnp.zeros((self._num_agents, 2)),
            'has_dribbled_yet': jnp.zeros(self._num_agents),
        }

        return State(data, obs, reward, done, metrics, info=info)

    def _post_init(self) -> None:
        """
        """
        pass

    def _get_reward(self,
                obs: jnp.ndarray,
                is_in_left_goal: bool, 
                is_in_right_goal: bool,
                dribbled_too_far_flag: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        def _get_dense_ball_to_goal_reward(ball_pos, ball_vel):
            """
            Computes a dense reward based on the ball's velocity towards a point
            inside the opponent's goal.

            This reward is positive if the ball is moving generally towards the target point,
            and zero otherwise (if stationary, moving away, or perpendicular).

            We approximate ball has to traverse 8.5 meters at max.
            At 60 hz control loop this can get up to 510 (unscaled).

            Args:
                ball_pos: The 3D position of the ball [x, y, z]
                ball_vel: The 3D velocity of the ball [vx, vy, vz]. Only the 2D
                          components (x, y) of the velocity are used.

            Returns:
                A scalar jnp.ndarray representing the component of the ball's 2D speed
                that is directed towards the center of the opponent's goal.
            """

            goal_depth_offset = 0.19 # This is directly at the geom
            target_goal_x = (self._field_width / 2) + goal_depth_offset
            target_goal_y = 0.0
            goal_pos = jnp.array([target_goal_x, target_goal_y, ball_pos[2]])

            ball_to_goal_vector = goal_pos[:2] - ball_pos[:2]
            ball_to_goal_distance = jnp.linalg.norm(ball_to_goal_vector) + 1e-10
            ball_to_goal_direction = ball_to_goal_vector / ball_to_goal_distance

            ball_vel_towards_goal = jnp.dot(ball_vel[:2], ball_to_goal_direction)

            # # --- DEBUG NANs in reward input ---
            # jax.debug.print("_get_dense_ball_to_goal_reward - ball_pos: {bp}, ball_vel: {bv}", bp=ball_pos, bv=ball_vel)
            # # --- END DEBUG ---

            is_ball_in_right_goal_flag = self._is_in_right_goal(ball_pos) # Returns 1.0 if in goal, 0.0 otherwise

            # Conditionally clip the reward
            final_dense_reward = jnp.where(
                is_ball_in_right_goal_flag > 0.5,       # True if is_ball_in_right_goal_flag is 1.0
                jnp.maximum(0., ball_vel_towards_goal), # If true
                ball_vel_towards_goal                   # If false
            )

            return final_dense_reward
        
        def _get_dense_base_to_ball_reward(robot_pos_2d: jnp.ndarray,
                                                    robot_vel_2d: jnp.ndarray,
                                                    robot_orientation_2d: jnp.ndarray,
                                                    ball_pos_2d: jnp.ndarray):
                """
                Computes dense reward for the robot moving towards the ball.
                Only gives reward if robot is far away (self.distance_threshold).

                Worst case scenario the robot has to traverse ~10 meters. If it does this at 1 m/s that's 9 seconds.
                Control loop is ~60hz, meaning it get's 60*9 = 540 for the reward.

                If it moves 0.5 m/s forward for 1 second. Thats 60 control loops at 0.5 = 30 * 0.02 (scaled) = 0.6
                Then lets say it moves back at 0.5 for 1 second. That costs 60*0.03 = 1.8
                Then it moves again 0.5 m/s forward for 1 second. Thats 0.6 again. But also costs 1.8 so 1.2 - 3.2 - 

                Args:
                    robot_pos_2d: position of the robot [x,y]
                    robot_vel_2d: velocity of the robot [x,y]
                    robot_orientation_2d: orientation in [cos, sin] (needed for direction calculation)
                    ball_pos_2d: position of the ball in [x,y]

                Returns:
                    Dense reward per step for moving to the ball if far away.
                """

                # Get direction and distance to the ball.
                direction_to_ball, distance_to_ball, _angle_rad = self._calculate_base_to_ball_score(
                    robot_pos_2d, robot_orientation_2d, ball_pos_2d
                )

                # Dot product of robot velocity vector and robot to ball vector
                robot_vel_towards_ball = jnp.dot(robot_vel_2d, direction_to_ball)

                # Reward for moving towards the ball, only if moving towards it (positive projection)
                move_towards_ball_reward = jnp.where(robot_vel_towards_ball > 0.0, robot_vel_towards_ball, 0.0)

                # Apply distance check: only give reward when far away
                distance_check = jnp.where(distance_to_ball > self.far_distance_threshold, 1.0, 0.0)
                move_towards_ball_reward = move_towards_ball_reward * distance_check
                
                return move_towards_ball_reward
            
        def _get_dense_facing_ball_reward(robot_pos_2d: jnp.ndarray,
                                                robot_orientation_2d: jnp.ndarray,
                                                ball_pos_2d: jnp.ndarray):
            """
            Computes dense reward for the robot facing the ball.

            Args:
                robot_pos_2d: position of the robot [x,y]
                robot_orientation_2d: orientation in [cos, sin]
                ball_pos_2d: position of the ball in [x,y]

            Returns:
                Dense reward per step for facing the ball.
            """

            # Get the angle to the ball
            _direction_to_ball, _distance_to_ball, angle_rad = self._calculate_base_to_ball_score(
                robot_pos_2d, robot_orientation_2d, ball_pos_2d
            )

            # Calculate the facing score using the angle
            facing_ball_score = self._calculate_robot_facing_score(angle_rad)
            
            return facing_ball_score
        
        def _get_single_reward(agent_obs, dribbled_too_far_flag: jnp.ndarray):
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

            # All dense rewards combined
            dense_ball_to_goal_reward = _get_dense_ball_to_goal_reward(ball_pos, ball_vel)
            dense_base_to_ball_reward = _get_dense_base_to_ball_reward(robot_pos, robot_vel, robot_orientation, ball_pos[:2]) # Pass only 2D ball position
            dense_ball_facing_reward = _get_dense_facing_ball_reward(robot_pos, robot_orientation, ball_pos[:2]) # Only pass 2D ball position

            # Now scale them:
            dense_ball_to_goal_reward_scaled = dense_ball_to_goal_reward * 0.5 # At 0.4 and ball 3 meters away, about 35.6
            dense_base_to_ball_reward_scaled = dense_base_to_ball_reward * 0.04 # At 0.4 and robot 0.75m reward, about 34
            dense_ball_facing_reward_scaled = dense_ball_facing_reward * 0.02 # Can reach 20 max.

            # 1, 0.4, 0.02

            # Logic here is that by doing nothing, the robot gets a negative reward of -30, just from existing.
            # If it goes to the ball, it gains about 2.4. It gets massively more 

            # # is_dribbling reward
            dribbling_bonus = jnp.where(is_dribbling > 0, 0.1, 0.0)
            dribbled_too_far_penalty = jnp.where(dribbled_too_far_flag > 0, -0.05, 0.0)
            
            # General rewards
            goal_reward = jnp.where(is_in_right_goal, 500.0, 0.0)
            out_of_bounds_penalty = jnp.where(is_out_of_bounds, -50.0, 0.0)
            time_penalty = -0.05
            
            total_reward = goal_reward + dense_ball_to_goal_reward_scaled + dense_base_to_ball_reward_scaled + dense_ball_facing_reward_scaled + out_of_bounds_penalty + time_penalty + dribbling_bonus
            
            reward_components = {
                'rew_goal': goal_reward,
                'rew_ball_to_goal': dense_ball_to_goal_reward_scaled,
                'rew_base_to_ball': dense_base_to_ball_reward_scaled,
                'rew_ball_facing': dense_ball_facing_reward_scaled,
                'rew_dribbling': dribbling_bonus,
                'pen_out_of_bounds': out_of_bounds_penalty,
                'pen_dribbling': dribbled_too_far_penalty
            }

            return total_reward, reward_components
        
        # 1. Unpack the two outputs of vmap
        all_total_rewards_array, all_components_pytree = jax.vmap(_get_single_reward)(
            agent_obs_batch, dribbled_too_far_flag
        )

        # 2. Sum only the array of total rewards
        final_total_reward = jnp.sum(all_total_rewards_array)

        # 3. Process the components PyTree for logging (extract scalar for single agent)
        if self._num_agents == 1:
            final_components_dict = jax.tree_map(lambda x: x[0], all_components_pytree)
        else:
            final_components_dict = jax.tree_map(jnp.sum, all_components_pytree)

        return final_total_reward, final_components_dict

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        
        # Action processing and physics
        is_dribbling_flag_prev = state.obs[15::self._obs_size]
        dribbler_actuator_signal = jnp.where(is_dribbling_flag_prev > 0.5, -1.0, 0.0)
        
        policy_vel_rot_actions = action[:3]
        final_vel_rot_actions = policy_vel_rot_actions * 2.0
        policy_kicker_action = action[3] * 1 # Not currently scaled
        
        kicker_trigger_threshold = 0.5
        is_attempting_kick_impulse = jnp.logical_and(
            policy_kicker_action > kicker_trigger_threshold,
            is_dribbling_flag_prev
        )
        pipeline_actions = jnp.concatenate([
            final_vel_rot_actions,
            dribbler_actuator_signal,
            is_attempting_kick_impulse
        ])
        
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, pipeline_actions)
        obs = self._get_obs(data)

        # Dribbling penalty logic
        is_dribbling_now_per_agent = obs[15::self._obs_size]
        current_robot_positions = obs.reshape(self._num_agents, self._obs_size)[:, 0:2]

        is_too_far, new_initial_dribble_pos, new_has_dribbled_yet, dribbling_distance = self._dribbled_too_far(
            current_robot_pos=current_robot_positions,
            is_dribbling_now=is_dribbling_now_per_agent,
            has_dribbled_yet_prev=state.info['has_dribbled_yet'],
            initial_dribble_pos_prev=state.info['initial_dribble_pos']
        )
        
        # Reward calculation
        is_out_of_bounds = obs[-4]
        is_in_left_goal = obs[-3]
        is_in_right_goal = obs[-2]
        reward, reward_components_dict = self._get_reward(obs, is_in_left_goal, is_in_right_goal, is_too_far)
        
        # Update metrics and state
        is_nan = jnp.logical_or(jnp.isnan(data.qpos).any(), jnp.isnan(data.qvel).any()).astype(jnp.float32)

        metrics = dict(state.metrics)
        metrics['out_of_bounds'] = is_out_of_bounds
        metrics['left_goal'] = is_in_left_goal
        metrics['right_goal'] = is_in_right_goal
        metrics['is_nan'] = is_nan
        metrics['attempted_kick_while_dribbling'] = jnp.any(is_attempting_kick_impulse).astype(jnp.float32)
        metrics['dribbling_distance_from_start'] = jnp.sum(dribbling_distance)
        
        is_dribbling_overall = jnp.any(is_dribbling_now_per_agent).astype(jnp.float32)
        metrics['is_dribbling'] = is_dribbling_overall
        
        latch_was_off_previously = (metrics['_internal_dribble_latch'] < 0.5)
        dribbling_active_now = (is_dribbling_overall > 0.5)
        metrics['did_dribble_this_episode'] = jnp.logical_and(dribbling_active_now, latch_was_off_previously).astype(jnp.float32)
        metrics['_internal_dribble_latch'] = jnp.maximum(metrics['_internal_dribble_latch'], is_dribbling_overall)
        
        for key, value in reward_components_dict.items():
            metrics[key] = value

        done = jnp.any(jnp.array([is_out_of_bounds, is_in_left_goal, is_in_right_goal, is_nan])).astype(jnp.float32)

        # Here's the change: Update the `info` dictionary with the new memory.
        new_info = state.info.copy()
        new_info['has_dribbled_yet'] = new_has_dribbled_yet
        new_info['initial_dribble_pos'] = new_initial_dribble_pos

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info
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

            ball_pos_x = data.qpos[self._ball_x_slide_qpos_adr]
            ball_pos_y = data.qpos[self._ball_y_slide_qpos_adr]
            ball_pos_z = self.ball_geom_z_pos
            ball_pos = jnp.array([ball_pos_x, ball_pos_y, ball_pos_z]) 
            # jax.debug.print("Generated ball position: x={x}, y={y}, z={z}", x=ball_pos_x, y=ball_pos_y, z=ball_pos_z)

            ball_vel_x = data.qvel[self._ball_x_slide_qvel_adr]
            ball_vel_y = data.qvel[self._ball_y_slide_qvel_adr]
            ball_vel = jnp.array([ball_vel_x, ball_vel_y, 0.0])
        
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
        x_abs = jnp.abs(x)
        y_abs = jnp.abs(y)

        field_half_width = self._field_width / 2   # 6.0
        field_half_height = self._field_height / 2 # 4.5
        goal_mouth_half_width = 0.9

        # Ball is out on top or bottom
        out_top_bottom_endlines = jnp.greater(y_abs, field_half_height)

        # Ball is out on either left or right side, and not in either of the goals.
        out_endlines_not_in_goal = jnp.logical_and(
            jnp.greater(x_abs, field_half_width),       # Beyond the x-limit of the field
            jnp.greater_equal(y_abs, goal_mouth_half_width) # And outside the y-span of the goal
        )

        return jnp.logical_or(out_top_bottom_endlines, out_endlines_not_in_goal).astype(jnp.float32)

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
            Check if the robot is dribbling based on the kicker_beam_rf rangefinder sensor.
            If there is no ball, sensor will output 1. If there is, the value will be lower.

            Args:
                data: MuJoCo simulation data of type mjx.Data.
                agent_id: The index of the agent.
                robot_vel: Velocity of the robot (x, y, z). Currently unused in this specific dribbling logic.
                
            Returns:
                A jnp.array([1.0]) if dribbling, jnp.array([0.0]) otherwise.
            """
            current_agent_sensor_id = self._kicker_beam_rf_ids[agent_id]
            
            # Check if the sensor ID is valid (e.g., not -1)
            is_sensor_id_valid = (current_agent_sensor_id != -1)

            def check_sensor_and_determine_dribbling(sensor_id_operand):
                """
                Called when sensor_id_operand is a valid sensor ID.
                Checks the rangefinder reading and returns dribbling status.
                """
                distance_value = data.sensordata[sensor_id_operand]
                # jax.debug.print("distance: x={distance}", distance=distance_value)
                is_beam_interrrupted = distance_value < 0.07

                return jax.lax.select(
                    is_beam_interrrupted,
                    jnp.array([1.0], dtype=jnp.float32),  # Dribbling
                    jnp.array([0.0], dtype=jnp.float32)   # Not dribbling
                )
            
            def handle_invalid_sensor(unused_sensor_id_operand):
                """
                Called if the sensor ID was not valid (e.g., -1).
                Robot is considered not dribbling.
                """
                return jnp.array([0.0], dtype=jnp.float32) # Not dribbling
            
            is_dribbling_status = jax.lax.cond(
                is_sensor_id_valid,                     # Condition: is the sensor ID valid?
                check_sensor_and_determine_dribbling,   # True branch
                handle_invalid_sensor,                  # False branch
                current_agent_sensor_id
            )

            return is_dribbling_status
    
    def _calculate_base_to_ball_score(self, robot_pos_2d: jnp.ndarray, robot_orientation_2d: jnp.ndarray, ball_pos_2d: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes kinematic metrics related to the robot's position and orientation relative to the ball.

        Args:
            robot_pos_2d: Robot's 2D position (x, y).
            robot_orientation_2d: Robot's 2D orientation vector (cos(angle), sin(angle)).
            ball_pos_2d: Ball's 2D position (x, y).

        Returns:
            A tuple containing:
                - direction_to_ball_normalized (jnp.ndarray): Normalized 2D vector from robot to ball.
                - distance_to_ball (jnp.ndarray): Scalar L2 distance from robot to ball.
                - angle_rad (jnp.ndarray): Angle in radians between robot's orientation and direction to ball.
        """

        robot_to_ball_vector_2d = ball_pos_2d - robot_pos_2d
        distance_to_ball = jnp.linalg.norm(robot_to_ball_vector_2d) + 1e-10 # Added 1e-10 for stability
        direction_to_ball_normalized = robot_to_ball_vector_2d / distance_to_ball

        dot_product = jnp.clip(jnp.dot(robot_orientation_2d, direction_to_ball_normalized), -1.0, 1.0)
        angle_rad = jnp.arccos(dot_product)  # Angle in radians

        # Return all three important metrics
        return direction_to_ball_normalized, distance_to_ball, angle_rad
    
    def _calculate_robot_facing_score(self, angle_rad: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the facing score based on the angle to the target.

        Args:
            angle_rad: Angle in radians between robot's orientation and direction to target.

        Returns:
            facing_score (jnp.ndarray): Scalar score (0-1) indicating how directly the robot faces the target.
        """
        # facing_score: closer to 1 if angle is small, closer to 0 if angle is large.
        facing_score_param = 0.4
        facing_score = jnp.exp(-(angle_rad / facing_score_param)**2)
        return facing_score

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
    
    def _dribbled_too_far(self,
                            current_robot_pos: jnp.ndarray,
                            is_dribbling_now: jnp.ndarray,
                            has_dribbled_yet_prev: jnp.ndarray,
                            initial_dribble_pos_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Checks if the robot has dribbled too far from its initial dribble point.
            Handles batched inputs for multi-agent support.
            """
            # Check if this is the very first time each agent is dribbling this episode
            is_first_dribble_event = jnp.logical_and(
                is_dribbling_now > 0.5,
                has_dribbled_yet_prev < 0.5
            )

            # Latch the initial dribble position for each agent that starts dribbling.
            # The [:, None] correctly broadcasts the condition for multi-agent case.
            new_initial_dribble_pos = jnp.where(
                is_first_dribble_event[:, None],
                current_robot_pos,
                initial_dribble_pos_prev
            )

            # Latch the 'has_dribbled_yet' flag. Once true, it stays true.
            new_has_dribbled_yet = jnp.maximum(has_dribbled_yet_prev, is_dribbling_now)

            # Calculate distance from the initial point, but only if a dribble has occurred.
            dribbling_distance = jnp.where(
                new_has_dribbled_yet > 0.5,
                jnp.linalg.norm(current_robot_pos - new_initial_dribble_pos, axis=-1),
                0.0
            )
            
            # Convert the distance check to a boolean flag for each agent.
            is_too_far = dribbling_distance > 2.0

            # MODIFIED: Return the distance as well for logging.
            return is_too_far, new_initial_dribble_pos, new_has_dribbled_yet, dribbling_distance
        
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
    
