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

        physics_steps_per_control_step = 5
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
                mj_model.geom(f"kicker_plate_{robot_id}").id
            )

            k_id = mj_model.geom(f"kicker_plate_{robot_id}").id
            self._kicker_geom_ids = self._kicker_geom_ids.at[robot_id].set(k_id)
            # print(f"DEBUG: Agent {robot_id} Kicker Geom ID = {k_id} (Name: {mj_model.geom(k_id).name})")
        
        print(f"DEBUG: Ball Geom ID = {self._ball_geom_id} (Name: {mj_model.geom(self._ball_geom_id).name})")
        print(f"DEBUG: Kicker Geom ID[0] = {self._kicker_geom_ids[0]}")
        print(f"DEBUG: Ball X slide qpos adr: {self._ball_x_slide_qpos_adr}")
        print(f"DEBUG: Ball Y slide qpos adr: {self._ball_y_slide_qpos_adr}")

        self.far_distance_threshold = 0.5

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

        # Define a small margin to avoid spawning exactly on the edge
        pos_margin = 1
        x_min = (-self._field_width / 2) + pos_margin
        x_max = (self._field_width / 2) - pos_margin
        y_min = (-self._field_height / 2) + pos_margin
        y_max = (self._field_height / 2) - pos_margin

        # Split RNG key for ball and agents
        rng, ball_x_rng = jax.random.split(rng)
        rng, ball_y_rng = jax.random.split(rng)

        # # Generate random positions for the ball
        ball_x = jax.random.uniform(ball_x_rng, minval=x_min, maxval=x_max)
        ball_y = jax.random.uniform(ball_y_rng, minval=y_min, maxval=y_max)

        # # Generate random positions for the ball
        # ball_x = -3
        # ball_y = 1

        # Ball is put on center of the field
        qpos = qpos.at[self._ball_x_slide_qpos_adr].set(ball_x)
        qpos = qpos.at[self._ball_y_slide_qpos_adr].set(ball_y)
        # jax.debug.print("Generated ball position: x={x}, y={y}", x=ball_x, y=ball_y)

        for robot_id in range(self._num_agents):
            # Split rng for this agent's x, y, and rotation
            rng, agent_x_rng = jax.random.split(rng)
            rng, agent_y_rng = jax.random.split(rng)
            rng, agent_rot_rng = jax.random.split(rng)

            x_pos = jax.random.uniform(agent_x_rng, minval=x_min, maxval=x_max)
            y_pos = jax.random.uniform(agent_y_rng, minval=y_min, maxval=y_max)
            rot_pos = jax.random.uniform(agent_rot_rng, minval=-jnp.pi, maxval=jnp.pi)

            qpos = qpos.at[self._x_qpos_adr[robot_id]].set(x_pos)
            qpos = qpos.at[self._y_qpos_adr[robot_id]].set(y_pos)
            qpos = qpos.at[self._z_qpos_adr[robot_id]].set(rot_pos)

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

        def _get_dense_ball_to_goal_reward(ball_pos, ball_vel):

            goal_pos = jnp.array([self._field_width / 2, 0.0, ball_pos[2]])

            ball_to_goal_vector = goal_pos[:2] - ball_pos[:2]
            ball_to_goal_distance = jnp.linalg.norm(ball_to_goal_vector) + 1e-10
            ball_to_goal_direction = ball_to_goal_vector / ball_to_goal_distance

            ball_vel_towards_goal = jnp.dot(ball_vel[:2], ball_to_goal_direction)
            return ball_vel_towards_goal * 25
        
        def _get_dense_base_to_ball_reward(robot_pos: jnp.ndarray,
                                                  robot_vel: jnp.ndarray,
                                                  robot_orientation: jnp.ndarray,
                                                  ball_pos: jnp.ndarray):
            # Call the refactored instance method
            direction_to_ball, facing_ball_score, distance_to_ball = self._calculate_robot_to_ball_metrics(robot_pos, robot_orientation, ball_pos)

            # Velocity of robot projected onto the direction vector pointing to the ball
            robot_vel_towards_ball = jnp.dot(robot_vel, direction_to_ball)

            # Reward for moving towards the ball, only if moving towards it (positive projection)
            move_towards_ball_reward = jnp.where(robot_vel_towards_ball > 0.0, robot_vel_towards_ball, 0.0)

            # Apply distance multiplier: reward is higher if the robot is far from the ball
            distance_multiplier = jnp.where(distance_to_ball > self.far_distance_threshold, 1.0, 0.0) # Uses self.far_distance_threshold
            move_towards_ball_reward_dist_modified = move_towards_ball_reward * distance_multiplier

            # Reward scaling factors (consider making these named constants or class attributes)
            move_reward_scale = 0.05
            facing_score_scale = 0.02
            dense_reward = (move_towards_ball_reward_dist_modified * move_reward_scale +
                            facing_ball_score * facing_score_scale)
            return dense_reward

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

            dense_ball_to_goal_reward = _get_dense_ball_to_goal_reward(ball_pos, ball_vel)
            dense_base_to_ball_reward = _get_dense_base_to_ball_reward(
                robot_pos,
                robot_vel,
                robot_orientation,
                ball_pos[:2]  # Pass only 2D ball position
            )
            
            # General rewards
            goal_reward = jnp.where(is_in_right_goal, 100.0, 0.0)
            out_of_bounds_penalty = jnp.where(is_out_of_bounds, -1.0, 0.0)
            
            total_reward = goal_reward + dense_ball_to_goal_reward + dense_base_to_ball_reward + out_of_bounds_penalty - 0.01
            
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

        # 4 actions, x_vel, y_vel, rot_vel, kicker

        # --- Process dribbling policy action ---
        is_dribbling_flag = state.obs[15::16]
        is_dribbler_on = jnp.where(is_dribbling_flag, 1.0, 0.0)

        # --- Process policy controlled actions (Velocity, Rotation, Kicker Impulse) ---
        # Input 'action' is shape [4]: [x_vel, y_vel, rot_vel, kicker]
        policy_vel_rot_actions = action[:3]
        final_vel_rot_actions = policy_vel_rot_actions * 3.0 # Scale velocity/rotation actions from policy range [-1, 1] to actuator ctrlrange [-3, 3]

        # --- Process kicker policy action ---
        policy_kicker_action = action[3] # This is a number between 0 and 1

        kicker_trigger_threshold = 0.5
        is_attempting_kick_impulse = jnp.logical_and(
            policy_kicker_action > kicker_trigger_threshold,
            is_dribbling_flag
        )

        # Combine everything into a single action array to put into pipeline_step
        # Based on the MuJoCo XML, the actuators are x_vel, y_vel, rot_vel, dribbler, kicker_force_actuator.
        pipeline_actions = jnp.concatenate([
            final_vel_rot_actions,                                        # [3] for x_vel, y_vel, rot_vel
            is_dribbler_on,         # [1] for dribbler (sticky dribbler) - activate with 1.0 when dribbling
            is_attempting_kick_impulse # [1] for kicker_force_actuator (impulse kick)
        ])
        
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, pipeline_actions)

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

        possible_done_conditions = jnp.array([
            is_out_of_bounds,
            is_in_left_goal,
            is_in_right_goal,
            is_nan,
        ])
        game_ending_condition = jnp.any(possible_done_conditions)

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

        ball_vel_x = data.qvel[self._ball_x_slide_qvel_adr]
        ball_vel_y = data.qvel[self._ball_y_slide_qvel_adr]
        ball_vel_xy_vector = jnp.array([ball_vel_x, ball_vel_y])

        # --- 1. Check for Kicker-Ball Contact (Keep the robust contact check) ---
        ncon = data.ncon
        kicker_id = self._kicker_geom_ids[agent_id]
        ball_id = self._ball_geom_id

        geom1 = data.contact.geom1 # 1D array of all collision points, populated by the nearest geom to the contact point
        geom2 = data.contact.geom2
        dist = data.contact.dist # 1D array of all distances to the nearest geom
        max_contacts = geom1.shape[0] # Total number of rows in the contact array of geom1

        mask_is_active = jnp.arange(max_contacts) < ncon
        mask_is_target_pair = ((geom1 == kicker_id) & (geom2 == ball_id) | (geom1 == ball_id) & (geom2 == kicker_id))
        mask_is_touching = (dist <= 0.01)

        found_actual_contact = jnp.any(mask_is_active & mask_is_target_pair & mask_is_touching)

        # --- 2. Check Ball Speed ---
        ball_vel_scalar = jnp.linalg.norm(ball_vel_xy_vector) # Ball velocity as a scalar
        ball_speed_threshold = 0.5
        is_ball_slow = ball_vel_scalar < ball_speed_threshold

        # --- 3. Check Relative Speed between Robot Base and Ball ---
        robot_vel_xy_vector = robot_vel[:2]
        
        # Check relative velocity
        relative_vel_xy_vector = ball_vel_xy_vector - robot_vel_xy_vector
        relative_vel_xy = jnp.linalg.norm(relative_vel_xy_vector)

        # Define a threshold for relative speed
        relative_speed_threshold = 0.2 # You'll need to tune this threshold
        is_relative_speed_low = relative_vel_xy < relative_speed_threshold

        # --- 4. Final Dribbling Check ---
        # Dribbling requires contact AND low ball speed AND low relative speed.
        is_dribbling = found_actual_contact

        # jax.debug.print("Agent {id}: contact={c}",
        #                 id=agent_id,
        #                 c=is_dribbling)
        return jnp.array([is_dribbling], dtype=jnp.float32)
    
    def _calculate_robot_to_ball_metrics(self, robot_pos_2d: jnp.ndarray, robot_orientation_2d: jnp.ndarray, ball_pos_2d: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes metrics related to the robot's position and orientation relative to the ball.

        Args:
            robot_pos_2d: Robot's 2D position (x, y).
            robot_orientation_2d: Robot's 2D orientation vector (cos(angle), sin(angle)).
            ball_pos_2d: Ball's 2D position (x, y).

        Returns:
            A tuple containing:
                - direction_to_ball (jnp.ndarray): Normalized 2D vector from robot to ball. Shape (2,).
                - facing_score (jnp.ndarray): Scalar score (0-1) indicating how directly the robot faces the ball. Shape ().
                - distance_to_ball (jnp.ndarray): Scalar L2 distance from robot to ball. Shape ().
        """

        robot_to_ball_vector_2d = ball_pos_2d - robot_pos_2d
        distance_to_ball = jnp.linalg.norm(robot_to_ball_vector_2d) + 1e-10
        direction_to_ball_normalized = robot_to_ball_vector_2d / distance_to_ball

        dot_product = jnp.clip(jnp.dot(robot_orientation_2d, direction_to_ball_normalized), -1.0, 1.0)
        angle_rad = jnp.arccos(dot_product)  # Angle in radians

        # facing_score: closer to 1 if angle is small, closer to 0 if angle is large.
        facing_score_param = 0.4
        facing_score = jnp.exp(-(angle_rad / facing_score_param)**2)

        return direction_to_ball_normalized, facing_score, distance_to_ball

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
    
