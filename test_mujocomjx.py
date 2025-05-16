import jax
from jax import numpy as jnp
import numpy as np # Import standard NumPy for array conversion
import mujoco
from mujoco import mjx
from brax.io import mjcf
import os # To check file existence
import time # For potential delay

# Ensure JAX operations are complete before NumPy conversion
def ensure_sync(arr):
    """Forces JAX computation to complete for the given array."""
    # Check if it's a JAX array before calling block_until_ready
    if hasattr(arr, 'block_until_ready'):
        arr.block_until_ready()
    return arr

def test_mjx_initialization_and_step(xml_path: str):
    """
    Loads a MuJoCo model, initializes qpos/qvel to zeros,
    calls mjx.make_data, sets the initial state, performs one mjx.step,
    and prints the resulting ncon along with the names of contacting geoms.
    """
    print(f"\n--- Attempting MJX Initialization and Step for: {xml_path} ---")

    # --- Explicit File Existence Check ---
    if not os.path.exists(xml_path):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: XML file NOT FOUND at the specified path: {xml_path}")
        print(f"       Current working directory: {os.getcwd()}")
        # Check common alternative location (assuming it might be inside robot_model)
        if not os.path.dirname(xml_path): # If it's just a filename
             alt_path = os.path.join("robot_model", xml_path)
             if os.path.exists(alt_path):
                  print(f"       Did you mean: {alt_path} ?")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Check for STL file existence if relevant
        if "robot" in xml_path and not os.path.exists("robot_decimated.stl"):
             # Check common location for STL too
             alt_stl_path = os.path.join(os.path.dirname(xml_path) if os.path.dirname(xml_path) else ".", "robot_decimated.stl")
             if not os.path.exists(alt_stl_path):
                print(f"Warning: 'robot_decimated.stl' not found relative to script ({os.getcwd()}) or XML ({alt_stl_path}). Ensure it's accessible.")
        return
    else:
        print(f"XML file found at: {xml_path}")


    try:
        # --- Model Loading ---
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"MuJoCo Model Loaded: nq={mj_model.nq}, nv={mj_model.nv}, ngeom={mj_model.ngeom}")
        mjx_model = mjx.put_model(mj_model)
        print(f"MJX Model Loaded.")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        if "Error opening file" in str(e) and ".stl" in str(e):
             print("\nHint: Ensure the STL file path inside the XML is correct")
             print(f"      and the STL file exists relative to: {os.getcwd()}")
        return

    # --- Initial State Setup ---
    initial_qpos = jnp.zeros(mjx_model.nq)
    initial_qvel = jnp.zeros(mjx_model.nv)
    print(f"Target initial qpos (zeros): shape={initial_qpos.shape}")
    print(f"Target initial qvel (zeros): shape={initial_qvel.shape}")

    try:
        # --- MJX Data Initialization ---
        mjx_data = mjx.make_data(mjx_model)
        print("mjx.make_data(mjx_model) executed successfully.")
        # mjx_data = mjx_data.replace(qpos=initial_qpos, qvel=initial_qvel)
        print("mjx_data updated with initial qpos/qvel.")

        # --- Perform one MJX step ---
        print("Performing mjx.step...")
        jit_step = jax.jit(mjx.step)
        mjx_data = jit_step(mjx_model, mjx_data)
        print("mjx.step executed successfully.")

        # --- Check Contacts AFTER the step ---
        # Ensure computation is finished before reading ncon
        ncon = ensure_sync(mjx_data.ncon)
        ncon_py = int(np.array(ncon)) # Convert scalar JAX array to Python int
        print(f"\n>>> RESULT: MJX ncon AFTER 1 step = {ncon_py}\n")

        # Print contact pairs using corrected logic
        if 0 < ncon_py:
             print(f"--- MJX Contact Pairs (After 1 step, printing {min(ncon_py, 20)}/{ncon_py}) ---")
             try:
                 # Ensure contact arrays are computed before conversion
                 geom1_data = ensure_sync(mjx_data.contact.geom1[:ncon_py])
                 geom2_data = ensure_sync(mjx_data.contact.geom2[:ncon_py])

                 # Convert slices to NumPy arrays on host
                 geom1_np = np.array(geom1_data)
                 geom2_np = np.array(geom2_data)
                 print(f"DEBUG: Raw geom1_np[:{min(ncon_py, 5)}] = {geom1_np[:min(ncon_py, 5)]}")
                 print(f"DEBUG: Raw geom2_np[:{min(ncon_py, 5)}] = {geom2_np[:min(ncon_py, 5)]}")
             except Exception as e_conv:
                 print(f"ERROR converting contact geom arrays to NumPy: {e_conv}")
                 return

             printed_count = 0
             for i in range(ncon_py):
                  if printed_count >= 20:
                      print("  ... (omitting further pairs)")
                      break
                  try:
                      g1_id = int(geom1_np[i])
                      g2_id = int(geom2_np[i])

                      #Lookup names using the integer IDs from the original mj_model
                      g1_name = mj_model.geom(g1_id).name if 0 <= g1_id < mj_model.ngeom else f"INVALID_ID_{g1_id}"
                      g2_name = mj_model.geom(g2_id).name if 0 <= g2_id < mj_model.ngeom else f"INVALID_ID_{g2_id}"
                      print(f"  Contact {i}: {g1_name} ({g1_id}) <-> {g2_name} ({g2_id})")
                      printed_count += 1
                  except IndexError as e_idx:
                      print(f"ERROR accessing contact index {i} or geom name: {e_idx}")
                  except Exception as e_print:
                      print(f"ERROR printing contact pair {i}: {e_print}")

             print("------------------------")
        elif ncon_py == 0:
             print("No contacts reported by MJX after step.")


    except Exception as e:
        print(f"ERROR during mjx.make_data, mjx.step, or contact checking: {e}")

# --- Run the specific test ---
if __name__ == "__main__":
    # --- Test Full Original Model using MJX with one step ---
    # Make sure this path points to your original file
    target_xml_file = "robot_model/multi_robot_soccer_generated.xml"
    print(f"\n === Running MJX init and step test on: {target_xml_file} === ")
    # Use the updated function name
    test_mjx_initialization_and_step(target_xml_file)

