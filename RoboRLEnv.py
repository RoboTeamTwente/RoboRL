"""Simple RoboRL environment."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

class RoboRLEnv(mjx_env.MjxEnv):
    """Simple RoboRL environment."""

    def __init__(self, xml_path="robot.xml", config: Optional[config_dict.ConfigDict] = None):
        if config is None:
            config = config_dict.create(
                ctrl_dt=0.01,
                sim_dt=0.01,
                episode_length=2000, # Roughly 20 seconds
                action_repeat=1
            )

        super().__init__(config)

        self._xml_path = xml_path

        # Make model, data, and renderer
        self._mj_model = mujoco.MjModel.from_xml_path(self.xml_path) # This is on CPU
        self._mj_data = mujoco.MjData(self._mj_model)

        # Get the correct starting index for the ball's free joint in qpos and qvel
        self.ball_joint_id = self._mj_model.joint("ball_joint").id
        self.ball_qpos_start = self._mj_model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel_start = self._mj_model.jnt_dofadr[self.ball_joint_id]

        # Set the position part (last 3 elements of the 7 free joint values)
        # Default quaternion is [1,0,0,0] (no rotation)
        self._mj_data.qpos[self.ball_qpos_start:self.ball_qpos_start+7] = [1, 0, 0, 0, 1, 0, 0]

        # Set robot position using joint names directly
        self._mj_data.joint("x_slide").qpos[0] = 0  # x position
        self._mj_data.joint("y_slide").qpos[0] = 1   # y position
        self._mj_data.joint("z_rotate").qpos[0] = 0  # rotation in radians

        # Verify
        print(f"Ball qpos: {self._mj_data.qpos[self.ball_qpos_start+4:self.ball_qpos_start+7]}")

        # Verify robot position
        print(f"Robot position: x={self._mj_data.joint('x_slide').qpos[0]}, " +
            f"y={self._mj_data.joint('y_slide').qpos[0]}, " +
            f"rotation={self._mj_data.joint('z_rotate').qpos[0]}")
        
        # Put them on GPU using mjx
        self._mjx_model = mjx.put_model(self._mj_model)
        self._mjx_data = mjx.put_data(self._mj_model, self._mj_data)

        # Field dimensions for out-of-bounds checking
        self._field_width = 12.0
        self._field_height = 9.0

        # Get geometries of robots and dribbler
        self._ball_geom_id = -1
        self._kicker_geom_id = -1

        for i in range(self._mj_model.ngeom):
            geom_name = self._mj_model.geom(i).name
            if geom_name == "golf_ball_geom":
                self._ball_geom_id = i
            elif geom_name == "kicker_plate_geom":
                self._kicker_geom_id = i

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment.
        
        Args:
            rng: A JAX random key.
            
        Returns:
            The initial state.
        """
        # Create initial positions - start with zeros
        initial_qpos = jnp.zeros(11)

        # Set robot position (x, y, rotation) at indices 0, 1, 2
        initial_qpos = initial_qpos.at[0:3].set(
            jnp.array([0.0, 1.0, 0.0])  # x=0, y=1, rotation=0
        )
            
        # Set ball quaternion (identity rotation)
        initial_qpos = initial_qpos.at[self.ball_qpos_start:self.ball_qpos_start+4].set(
            jnp.array([1.0, 0.0, 0.0, 0.0])
        )
        
        # Set ball position
        initial_qpos = initial_qpos.at[self.ball_qpos_start+4:self.ball_qpos_start+7].set(
            jnp.array([1.0, 0.0, 0.1])
        )
        
        # Create the initial state with proper positions
        data = mjx_env.init(self.mjx_model, qpos=initial_qpos)

        # Initialize metrics and info
        metrics = {
            'out_of_bounds': jnp.zeros((), dtype=jnp.float32),
            'left_goal': jnp.zeros((), dtype=jnp.float32),
            'right_goal': jnp.zeros((), dtype=jnp.float32),
            'is_dribbling': jnp.zeros((), dtype=jnp.float32),
            'is_nan': jnp.zeros((), dtype=jnp.float32)
        }
        info = {
            'rng': rng,
            'out_of_bounds': jnp.zeros((), dtype=jnp.float32),
            'left_goal': jnp.zeros((), dtype=jnp.float32),
            'right_goal': jnp.zeros((), dtype=jnp.float32),
            'is_dribbling': jnp.zeros((), dtype=jnp.float32),
            'is_nan': jnp.zeros((), dtype=jnp.float32)
        }

        # Initial reward and done state
        reward = jnp.zeros((), dtype=jnp.float32)
        done = jnp.zeros((), dtype=jnp.float32)

        # Get initial observation
        obs = self._get_obs(data, info)

        # Create and return initial state
        return mjx_env.State(
            data=data, 
            obs=obs, 
            reward=reward, 
            done=done, 
            metrics=metrics, 
            info=info
        )

    def _get_reward(self, 
                    data: mjx.Data, 
                    action: jax.Array, 
                    is_out_of_bounds: bool, 
                    is_in_left_goal: bool, 
                    is_in_right_goal: bool, 
                    is_dribbling: bool) -> jax.Array:
        """
        Compute reward function.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            action: Action to take with type jax.Array.
            is_out_of_bounds: Flag indicating if ball is out of bounds
            is_in_left_goal: Flag indicating if ball is in left goal
            is_in_right_goal: Flag indicating if ball is in right goal

        Returns:
            Reward value as a JAX array.
        """

        # # Reward structure
        # goal_reward = jnp.where(is_in_right_goal, 10.0, 0.0)
        # out_of_bounds_penalty = jnp.where(is_out_of_bounds, -5.0, 0.0)
        # action_penalty = 0.1 * jnp.sum(jnp.square(action))

        # return goal_reward - action_penalty + out_of_bounds_penalty

        # Robot velocity
        robot_vel = data.qvel[:2]

        # Calculate base to ball velocity using vector (dot product)
        base_to_ball_vel = jnp.dot(self._get_base_to_ball_vector(data), robot_vel)

        # Positive rewards
        #dribbling_reward = jnp.where(is_dribbling, 1, 0.0)
        base_to_ball_reward = jnp.where(base_to_ball_vel > 0.0, jnp.abs(base_to_ball_vel), 0.0)
        facing_ball_reward = self._get_base_to_ball_angle_score(data)

        # Negative rewards
        out_of_bounds_penalty = jnp.where(is_out_of_bounds, -1.0, 0.0)
        action_penalty = 1
        
        # Return and apply scaling
        total_reward = base_to_ball_reward * 0.5 - out_of_bounds_penalty * 1 + facing_ball_reward * 0.025
    
        return total_reward

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        """
        Args:
        state: State of the environment with type mjx_env.State.
        action: Action to take with type jax.Array.

        Returns:
        state: Updated state of the environment with type mjx_env.State.
        """

        # Scale the incoming actions from [-1, 1] to [-3, 3]
        scaled_action = action * 3.0

        # Create full action vector with zeros for unused actuators
        full_action = jnp.zeros(self.mjx_model.nu)
        full_action = full_action.at[:3].set(scaled_action)  # Set only the first 3
        
        # Use full_action for simulation
        data = mjx_env.step(self.mjx_model, state.data, full_action, self.n_substeps)
        
        # Observation from within the simulation. obs is an jax.array
        obs = self._get_obs(data, state.info) # .info is unused

        # Extract condition values from observation
        is_out_of_bounds = obs[-4]
        is_in_left_goal = obs[-3]
        is_in_right_goal = obs[-2]
        is_dribbling = obs[-1]

        # Get reward
        reward = self._get_reward(data, action, is_out_of_bounds, is_in_left_goal, is_in_right_goal, is_dribbling)

        # Check for NaN values
        is_nan = jnp.logical_or(
            jnp.isnan(data.qpos).any(),
            jnp.isnan(data.qvel).any()
        ).astype(jnp.float32)

        #Check for game ending conditions
        game_ending_condition = jnp.any(jnp.array([
            is_out_of_bounds,  # Ball out of bounds
            is_in_left_goal,   # Ball in left goal
            is_in_right_goal,  # Ball in right goal
            is_nan,            # NaN values detected
            # is_dribbling == True  # Terminate when robot is dribbling
        ]))

        # Update info and metrics
        info = dict(state.info)
        info['out_of_bounds'] = is_out_of_bounds
        info['left_goal'] = is_in_left_goal
        info['right_goal'] = is_in_right_goal
        info['is_dribbling'] = is_dribbling
        info['is_nan'] = is_nan
        
        metrics = dict(state.metrics)
        metrics['out_of_bounds'] = is_out_of_bounds
        metrics['left_goal'] = is_in_left_goal
        metrics['right_goal'] = is_in_right_goal
        metrics['is_dribbling'] = is_dribbling
        metrics['is_nan'] = is_nan

        done = game_ending_condition.astype(jnp.float32)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """
        Args:
            data: MuJoCo simulation data.
            info: Additional info dict (unused).
            
        Returns:
            Observation array containing robot and ball state information.
        """

        del info  # Unused.
        
        # Get the robot state
        robot_state = self._get_robot_state(data, 1)  # Robot ID 1
        robot_pos = robot_state[0:2]
        robot_vel = robot_state[4:6]
        robot_orientation = robot_state[2:4]

        # Access ball position (after quaternion) and ball velocity (x, y, z)
        ball_pos = data.qpos[self.ball_qpos_start+4:self.ball_qpos_start+7]
        ball_vel = data.qvel[self.ball_qvel_start+3:self.ball_qvel_start+6]

        # Check conditions
        is_out_of_bounds = self._is_out_of_bounds(ball_pos)
        is_in_left_goal = self._is_in_left_goal(ball_pos)
        is_in_right_goal = self._is_in_right_goal(ball_pos)
        is_dribbling = self._is_dribbling(data, 1) # Just check the first robot

        return jnp.concatenate([
            robot_pos,                                          # Robot position (x, y)
            robot_vel,                                          # Robot velocity (x, y)
            robot_orientation,                                  # Robot orientation (cos, sin)
            ball_pos,                                           # Ball position (x, y, z)   
            ball_vel,                                           # Ball velocity (x, y, z)
            jnp.array([is_out_of_bounds]),       
            jnp.array([is_in_left_goal]),        
            jnp.array([is_in_right_goal]),       
            jnp.array([is_dribbling])                           # is dribbling
        ])

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
    
    def _is_dribbling(self, data: mjx.Data, robot_id: int) -> jax.Array:
        """
        Check if the robot is dribbling with the ball at low velocity.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            
        Returns:
            Boolean indicating whether the robot is dribbling with the ball.
        """

        # Get robot state which includes velocity
        robot_state = self._get_robot_state(data, robot_id)

        # Get velocity xy and speed magnitude
        robot_vel = robot_state[4:7]  # velocities of x_slide, y_slide and z_rotate
        robot_speed = jnp.sqrt(jnp.sum(jnp.square(robot_vel))) # Squares the values in robot_vel and sums them, then take sqrt for the magnitude

        # Check if robot is moving slowly
        is_low_speed = robot_speed < 0.2

        # Number of contacts in the simulation
        number_of_contacts = data.ncon

        # Initialize no_contacts_result to zero
        no_contacts_result = jnp.zeros((), dtype=jnp.bool_)
        
        # Get ball and kicker geom IDs
        ball_geom_id = self._ball_geom_id
        kicker_geom_id = self._kicker_geom_id
        
        # In MJX, contact data is stored in separate arrays
        geom1 = data.contact.geom1  # Array of first geom IDs for all contacts
        geom2 = data.contact.geom2  # Array of second geom IDs for all contacts
        
        # Check if any contact is between ball and kicker
        ball_kicker_contact_1 = jnp.logical_and(
            geom1 == ball_geom_id, 
            geom2 == kicker_geom_id
        )
        
        # Other way around
        ball_kicker_contact_2 = jnp.logical_and(
            geom2 == ball_geom_id,
            geom1 == kicker_geom_id
        )
        
        # Check for ball contact
        has_ball_contact = jnp.any(jnp.logical_or(ball_kicker_contact_1, ball_kicker_contact_2)) 

        has_contacts_result = jnp.logical_and(is_low_speed, has_ball_contact)

        # Return True if any contact is between ball and kicker and if the robot is moving slowly
        return jnp.where(number_of_contacts > 0, has_contacts_result, no_contacts_result).astype(jnp.float32)
    
    def _get_base_to_ball_vector(self, data: mjx.Data) -> jax.Array:
        """
        Function that computes the base to ball vector.

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            
        Returns:
            Vector from the robot base to the ball.
        """
        # Get the robot state
        robot_state = self._get_robot_state(data, 1)  # Robot ID 1
        robot_pos = robot_state[0:2] # x, y

        # Access ball position (after quaternion) (x, y, z)
        ball_pos = data.qpos[self.ball_qpos_start+4:self.ball_qpos_start+7]

        # Robot to ball vector
        robot_to_ball_vector = ball_pos[:2] - robot_pos # pointing from robot to ball
        distance_L2 = jnp.sqrt(jnp.sum(jnp.square(robot_to_ball_vector))) + 1e-10
        robot_to_ball_vector_unit = robot_to_ball_vector / distance_L2

        return robot_to_ball_vector_unit

    def _get_base_to_ball_angle_score(self, data: mjx.Data) -> jax.Array:
        """
        Function to give a score for how much the robot is facing the ball

        Args:
            data: MuJoCo simulation data of type mjx.Data.
            
        Returns:
            jax.Array: Score for how much the robot is facing the ball.
        """

        # Get robot orientation from robot state
        robot_state = self._get_robot_state(data, 1)  # Robot ID 1
        robot_orientation = robot_state[2:4]  # cos, sin

        # Get base to ball vector
        base_to_ball_vector = self._get_base_to_ball_vector(data)

        # Take dot product of robot orientation and ball vector
        dot_product = jnp.clip(jnp.dot(robot_orientation, base_to_ball_vector), -1.0, 1.0)

        # Calculate the angle difference, using dot product
        angle = jnp.arccos(dot_product)

        # Get score using the angle difference and formula from the MARLadona paper
        score = jnp.exp(-(angle/0.4)**2)

        return score

    def _get_robot_state(self, data: mjx.Data, robot_id: int) -> jax.Array:
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

        # Calculate starting indices
        # Assuming robots start at index 0 and each has 3 position values (x, y, rotation)
        pos_start = (robot_id - 1) * 3
        vel_start = (robot_id - 1) * 3  # Velocity indices match position indices
        
        # Robot position (x, y)
        robot_pos = data.qpos[pos_start:pos_start+2]
        
        # Robot rotation
        robot_rotation = data.qpos[pos_start+2]
        
        # Robot orientation as unit vector
        robot_orientation = jnp.array([jnp.cos(robot_rotation), jnp.sin(robot_rotation)])
        
        # Robot velocity
        robot_vel = data.qvel[vel_start:vel_start+3]
        
        # Return combined state
        return jnp.concatenate([robot_pos, robot_orientation, robot_vel])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 3 # 3 cause x_slide, y_slide, rotate

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model