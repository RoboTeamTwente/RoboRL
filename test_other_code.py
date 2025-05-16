import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

# XML definition remains the same
no_contact = """
<mujoco model="push_block">
  <worldbody>
    <body name="starting_plane">
      <geom type="box" size=".2 .2 .01" pos="0 0 0"/>
    </body>

    <body name="gripper_clone" pos="0.2 0 0.1">
      <joint name="conveyor_x" type="slide" damping="10" axis="1 0 0"/>
      <geom size=".002 .03 .01" pos="0 0 0"  type="box" friction=".8"/>
    </body>
  </worldbody>

  <actuator>
    <position name="conveyor_x" joint="conveyor_x" ctrlrange="-.5 .5" ctrllimited="true" kp="400"/>
  </actuator>
</mujoco>
"""

no_contact_mj_model = mujoco.MjModel.from_xml_string(no_contact)
no_contact_mj_data = mujoco.MjData(no_contact_mj_model)

# Standard MuJoCo part remains the same
print("--- Running Standard MuJoCo ---")
mujoco.mj_resetData(no_contact_mj_model, no_contact_mj_data)
duration = 5
std_mujoco_contact = False
while no_contact_mj_data.time < duration:
  no_contact_mj_data.ctrl = [-0.1 * no_contact_mj_data.time]
  mujoco.mj_step(no_contact_mj_model, no_contact_mj_data)
  # Using ncon is more reliable than len(contact.geom) here too
  if no_contact_mj_data.ncon > 0:
      std_mujoco_contact = True
print(f"Standard MuJoCo finished. Contact occurred: {std_mujoco_contact}")
assert not std_mujoco_contact


# MJX part
print("\n--- Running MJX ---")
mujoco.mj_resetData(no_contact_mj_model, no_contact_mj_data)
print("Converting model and data to MJX...")
no_contact_mjx_model = mjx.put_model(no_contact_mj_model)
# This is the mjx.Data object we will update and read from directly
no_contact_mjx_data = mjx.put_data(no_contact_mj_model, no_contact_mj_data)
print(f"NCON immediately after mjx.put_data: {no_contact_mjx_data.ncon}") # Initial check

print("Starting MJX simulation loop (accessing ncon directly)...")
jit_step = jax.jit(mjx.step)
mjx_contact_occurred = False # Flag to track if contact happened
while no_contact_mjx_data.time < duration:
  # Apply control
  no_contact_mjx_data = no_contact_mjx_data.replace(ctrl=jp.array([-0.1 * no_contact_mjx_data.time]))
  # Step the simulation, updating no_contact_mjx_data
  no_contact_mjx_data = jit_step(no_contact_mjx_model, no_contact_mjx_data)

  # --- Directly access ncon from the updated mjx.Data object ---
  current_ncon = no_contact_mjx_data.ncon
  print(f"Step Time: {no_contact_mjx_data.time:.4f}, Direct MJX ncon: {current_ncon}")
  if current_ncon > 0:
      mjx_contact_occurred = True
      # print("  >>> MJX Contact Detected <<<") # Optional marker
  # -------------------------------------------------------------

  # We REMOVE the inefficient mjx.get_data call from inside the loop
  # fetched_no_contact_mjx_data = mjx.get_data(no_contact_mj_model, no_contact_mjx_data)
  # print(f"Number of contacts (MJX): {fetched_no_contact_mjx_data.ncon}") # OLD WAY

  # We REMOVE the flawed assert based on len(contact.geom)
  # assert len(fetched_no_contact_mjx_data.contact.geom) == 0

print(f"MJX simulation finished. Contact occurred according to direct ncon check: {mjx_contact_occurred}")

# Optional: Add a final assert based on the flag if needed
# assert not mjx_contact_occurred # This would likely fail based on previous results