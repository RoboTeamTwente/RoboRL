import mujoco
from mujoco import mjx
import jax
import os

# Optional: Ensure float64 if your environment uses it
# jax.config.update("jax_enable_x64", True)
# print(f"JAX using float64: {jax.config.jax_enable_x64}")

xml_path = "robot_model/multi_robot_soccer_generated_simplified.xml" # Use the same XML as your env

if not os.path.exists(xml_path):
    print(f"ERROR: XML file not found at {xml_path}")
else:
    try:
        print("Loading with standard mujoco...")
        mj_model_std = mujoco.MjModel.from_xml_path(xml_path)
        print(f"  Standard mj_model_std.nconmax: {mj_model_std.ncon}")
        # if mj_model_std.nconmax <= 0: raise ValueError("Standard load failed!")

        # Optional: Apply the same options? Try commenting this block out first
        # print("Applying options...")
        # mj_model_std.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        # mj_model_std.opt.iterations = 100
        # mj_model_std.opt.ls_iterations = 100

        print("\nConverting directly to MJX model using mjx.put_model...")
        mjx_model = mjx.put_model(mj_model_std)
        print("  Direct MJX conversion successful.")
        print(f"  Direct mjx_model.nconmax: {mjx_model.nconmax}") # *** CHECK THIS ***
        print(f"  Direct mjx_model.ngeom: {mjx_model.ngeom}")
        print(f"  Direct mjx_model.nq: {mjx_model.nq}")
        print(f"  Direct mjx_model.nv: {mjx_model.nv}")

        # Optional: Try creating data
        # print("\nCreating standard mjData...")
        # mj_data_std = mujoco.MjData(mj_model_std)
        # print("Converting data with mjx.put_data...")
        # mjx_data = mjx.put_data(mj_model_std, mj_data_std)
        # print("  Direct MJX data conversion successful.")
        # print(f"  Initial mjx_data.ncon: {mjx_data.ncon}") # Check ncon after conversion

    except Exception as e:
        print(f"\nERROR during script execution: {e}")
        import traceback
        traceback.print_exc()