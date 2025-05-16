import numpy as np
import mujoco
import os
import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx

jax.config.update("jax_enable_x64", True)
print(f"JAX using float64: {jax.config.jax_enable_x64}")

def test_mujoco_initialization(xml_path: str):
    """
    Loads a MuJoCo model, initializes qpos/qvel to zeros using standard MuJoCo,
    runs mj_forward, and prints the resulting ncon and contact pairs.
    """
    print(f"\n--- Testing Standard MuJoCo Initialization for: {xml_path} ---")

    # --- File Existence Check ---
    if not os.path.exists(xml_path):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: XML file NOT FOUND at the specified path: {xml_path}")
        print(f"       Current working directory: {os.getcwd()}")
        alt_path = os.path.join("robot_model", os.path.basename(xml_path))
        if os.path.exists(alt_path):
             print(f"       Did you mean: {alt_path} ?")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if "robot" in xml_path and not os.path.exists("robot_decimated.stl"):
             print(f"Warning: 'robot_decimated.stl' not found in CWD ({os.getcwd()}). Ensure it's accessible if needed by the XML.")
        return
    else:
        print(f"XML file found at: {xml_path}")

    try:
        # --- Load Standard MuJoCo Model and Data ---
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"MuJoCo Model Loaded: nq={mj_model.nq}, nv={mj_model.nv}, ngeom={mj_model.ngeom}")
        mj_data = mujoco.MjData(mj_model)
        print("mujoco.MjData created.")

    except Exception as e:
        print(f"ERROR loading model or creating data: {e}")
        if "Error opening file" in str(e) and ".stl" in str(e):
             print("\nHint: Ensure the STL file path inside the XML is correct")
             print(f"      and the STL file exists relative to: {os.getcwd()}")
        return

    # --- Set Initial State (Zeros) ---
    # initial_qpos = np.zeros(mj_model.nq)
    # initial_qvel = np.zeros(mj_model.nv)
    # print(f"Target initial qpos (zeros): shape={initial_qpos.shape}")
    # print(f"Target initial qvel (zeros): shape={initial_qvel.shape}")

    # # Assign state directly to mj_data fields
    # mj_data.qpos[:] = initial_qpos
    # mj_data.qvel[:] = initial_qvel
    # print("Initial qpos/qvel assigned to mj_data.")

    mujoco.mj_resetData(mj_model, mj_data)
    duration = 5
    while mj_data.time < duration:
        mj_data.ctrl = [-0.1 * mj_data.time]
        mujoco.mj_step(mj_model, mj_data)

    # assert len(mj_data.contact.geom) == 0
    print(f"Number of contacts: {mj_data.ncon}")
    print("No errors, now test MJX")

    print("Converting to MJX...")
    no_contact_mjx_model = mjx.put_model(mj_model)
    no_contact_mjx_data = mjx.put_data(mj_model, mj_data) # Use the zeroed mj_data
    # print(f"Number of contacts (MJX) after put_data with zeroed state: {no_contact_mjx_data.ncon}")

    mujoco.mj_resetData(mj_model, mj_data)
    no_contact_mjx_model = mjx.put_model(mj_model)
    no_contact_mjx_data = mjx.put_data(mj_model, mj_data)
    # assert len(no_contact_mjx_data.contact.geom) == 8
    print(f"Number of contacts (MJX): {no_contact_mjx_data.contact.dist}")

    jit_step = jax.jit(mjx.step)
    while no_contact_mjx_data.time < duration:
        control_value = -0.1 * no_contact_mjx_data.time
        no_contact_mjx_data = no_contact_mjx_data.replace(ctrl=jp.array([control_value, control_value, control_value, control_value]))
        no_contact_mjx_data = jit_step(no_contact_mjx_model, no_contact_mjx_data)

        fetched_no_contact_mjx_data = mjx.get_data(mj_model, no_contact_mjx_data)
        # assert len(fetched_no_contact_mjx_data.contact.geom) == 0
        # print(f"Number of contacts (MJX): {fetched_no_contact_mjx_data.ncon}")
        print(f"Number of contacts (MJX): {no_contact_mjx_data.contact.dist}")
        
# --- Run the specific test ---
if __name__ == "__main__":
    # --- Test Robot (Mesh) + Kicker using Standard MuJoCo ---
    # Make sure this filename matches the XML file you used in the last MJX test
    # that gave ncon=8 with invalid IDs.
    target_xml_file = "robot_model/multi_robot_soccer_generated_simplified.xml" # <-- ADJUST FILENAME IF NEEDED
    print(f"\n === Running Standard MuJoCo test on: {target_xml_file} === ")
    test_mujoco_initialization(target_xml_file)
