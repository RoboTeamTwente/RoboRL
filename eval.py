import jax
import jax.numpy as jnp # Make sure jnp is imported
from brax.io import model # Make sure model loading is imported
from RoboRLEnv import RoboRLEnv
from customPPO import customNetworksPPO as ppo_networks # Your custom PPO
import mediapy as media
import numpy as np # For image saving if needed
import traceback # For printing errors

# --- Optional: Enable float64 for potentially better numerical stability ---
# Place this near the top of your script
jax.config.update("jax_enable_x64", True)
print(f"JAX using float64: {jax.config.jax_enable_x64}")
# -----------------------------------------------------------------------

# --- Your existing setup code ---
num_agents = 1
obs_size_per_agent = 16
env = RoboRLEnv(num_agents=num_agents)
obs_size = env.observation_size # No need to multiply by num_agents if already done in env
action_size = env.action_size # No need to multiply by num_agents if already done in env

ppo_networks_module = ppo_networks.make_ppo_networks(
    num_agents=num_agents,
    agent_observation_size=obs_size_per_agent,
    agent_action_size=4, # action size per agent
    policy_hidden_layer_sizes=(32,) * 4,
    value_hidden_layer_sizes=(256,) * 5,
)

model_path = 'mjx_brax_policy'
print("Loading model from: ", model_path)
params = model.load_params(model_path)

# --- Create policy function ---
make_policy = ppo_networks.make_inference_fn(ppo_networks_module, num_agents, obs_size_per_agent)

# --- Define JITted and NON-JITTED versions ---
# JITted versions (for the main video rollout later)
jit_inference_fn = jax.jit(make_policy(params))
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# NON-JITTED versions (for the initial debug loop)
print("Creating non-JIT versions for debug...")
inference_fn = make_policy(params) # Non-jitted policy
step_fn = env.step           # Non-jitted step
reset_fn = env.reset         # Non-jitted reset

# --- Get necessary indices ---
# Ensure these attributes exist and are correct in your RoboRLEnv
try:
    ball_pos_start_index = env._ball_qpos_start + 4
    ball_pos_end_index = env._ball_qpos_start + 7
    ball_vel_start_index = env._ball_qvel_start + 3
    ball_vel_end_index = env._ball_qvel_start + 6
except AttributeError as e:
    print(f"ERROR: Could not get ball index attributes from environment: {e}")
    print("Ensure _ball_qpos_start and _ball_qvel_start are set in RoboRLEnv.__init__")
    exit() # Exit if indices are missing

eval_env = env # Use the same env instance

# ==============================================================
# START: NON-JITTED DEBUG LOOP FOR FIRST FEW STEPS
# ==============================================================
print("\n--- Running Non-JIT Debug Loop (First 5 Steps) ---")

# Reset environment (use non-jitted version for full debug visibility)
rng_debug = jax.random.PRNGKey(3) # Use same seed for consistency
state_debug = reset_fn(rng_debug) # Use NON-JITTED reset

initial_ball_pos = state_debug.pipeline_state.qpos[ball_pos_start_index:ball_pos_end_index]
initial_ball_vel = state_debug.pipeline_state.qvel[ball_vel_start_index:ball_vel_end_index]
print(f"Initial State (after reset): Ball Pos = {initial_ball_pos}")
print(f"Initial State (after reset): Ball Vel = {initial_ball_vel}")

rollout_debug_states = [state_debug] # Store full State tuple
MAX_DEBUG_STEPS = 5

for i in range(MAX_DEBUG_STEPS):
    print(f"\n----- Debug Step {i} -----")
    current_state_debug = rollout_debug_states[-1] # Get the latest state
    qpos_before = current_state_debug.pipeline_state.qpos
    qvel_before = current_state_debug.pipeline_state.qvel
    print(f"State BEFORE step {i}: Ball Pos = {qpos_before[ball_pos_start_index:ball_pos_end_index]}")
    print(f"State BEFORE step {i}: Ball Vel = {qvel_before[ball_vel_start_index:ball_vel_end_index]}")

    # Get Action
    act_rng_debug, rng_debug = jax.random.split(rng_debug)
    # Ensure obs_input shape matches policy expectation
    # Assuming policy expects (batch, obs_dim) -> (1, num_agents * obs_per_agent)
    obs_input = current_state_debug.obs.reshape(1, -1)
    # Use non-jitted policy
    ctrl, _ = inference_fn(obs_input, act_rng_debug)
    # Ensure action shape matches step expectation
    # Assuming ctrl is (batch, action_dim) and step wants (action_dim)
    action_input = ctrl.squeeze(0)
    print(f"Action Input step {i} (shape {action_input.shape}) = {action_input}")

    # Take Step (Non-JITted)
    try:
        next_state_debug = step_fn(current_state_debug, action_input) # Use NON-JITTED step
        rollout_debug_states.append(next_state_debug) # Store next state

        # Analyze Result
        qpos_after = next_state_debug.pipeline_state.qpos
        qvel_after = next_state_debug.pipeline_state.qvel
        print(f"State AFTER step {i}: Ball Pos = {qpos_after[ball_pos_start_index:ball_pos_end_index]}")
        print(f"State AFTER step {i}: Ball Vel = {qvel_after[ball_vel_start_index:ball_vel_end_index]}")
        print(f"State AFTER step {i}: Done = {next_state_debug.done}")

        # Check for NaN/Inf explicitly
        nan_qpos = jnp.isnan(qpos_after).any()
        inf_qpos = jnp.isinf(qpos_after).any()
        nan_qvel = jnp.isnan(qvel_after).any()
        inf_qvel = jnp.isinf(qvel_after).any()
        if nan_qpos or inf_qpos or nan_qvel or inf_qvel:
             print(f"!!! WARNING step {i}: qpos NaN={nan_qpos}, Inf={inf_qpos}; qvel NaN={nan_qvel}, Inf={inf_qvel} !!!")
             break # Stop if state becomes invalid

        if next_state_debug.done.any():
            print(f"Termination condition met at debug step {i}")
            break
    except Exception as e:
        print(f"!!! ERROR during step {i}: {e} !!!")
        traceback.print_exc()
        break

print("\n--- End Debug Loop ---")

# ==============================================================
# END: NON-JITTED DEBUG LOOP
# ==============================================================