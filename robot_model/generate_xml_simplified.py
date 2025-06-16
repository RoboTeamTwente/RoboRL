import textwrap
import os # Added to ensure directory exists

# --- Configuration ---
NUM_AGENTS_TEAM_A = 1 # Example: 1 agent in team A
NUM_AGENTS_TEAM_B = 0 # Example: 0 agents in team B
TOTAL_AGENTS = NUM_AGENTS_TEAM_A + NUM_AGENTS_TEAM_B

# Define starting positions and colors
# Using the configuration from the target XML (Agent 0, blue, pos -2 0 0)
AGENT_CONFIG = [
    # Team A
    {'id': 0, 'pos': "-2 0 0", 'rgba': "0.2 0.2 0.8 1"}, # Match base color from target XML
    # Add more agents here if needed, ensure IDs are unique and sequential from 0
]

# Ensure AGENT_CONFIG matches TOTAL_AGENTS
if len(AGENT_CONFIG) != TOTAL_AGENTS:
    raise ValueError(f"AGENT_CONFIG length ({len(AGENT_CONFIG)}) does not match TOTAL_AGENTS ({TOTAL_AGENTS})")

# Output file path
output_dir = "robot_model"
output_filename = os.path.join(output_dir, "multi_robot_soccer_generated_simplified.xml")

# === Helper Functions to Generate XML Parts ===

def get_robot_body(agent_id: int, pos: str, rgba: str) -> str:
    """Generates the XML <worldbody> definition for a single robot using box primitives."""
    # Define robot dimensions (half-sizes) - same as target XML
    base_half_depth = 0.08
    base_half_width = 0.10
    base_half_height = 0.018
    kicker_half_thick = 0.005
    kicker_half_width = 0.0665 # Approx 2/3 of base width
    kicker_half_height = 0.018

    # Calculate positions based on dimensions
    base_pos_z = base_half_height
    kicker_body_pos_x = base_half_depth
    kicker_body_pos_z = base_half_height
    kicker_geom_pos_x = kicker_half_thick

    return textwrap.dedent(f"""
    <body name="robot_planar_{agent_id}" pos="{pos}">
        <joint name="x_slide_{agent_id}" type="slide" axis="1 0 0" damping="1.0" armature="0.1" limited="false"/>
        <joint name="y_slide_{agent_id}" type="slide" axis="0 1 0" damping="1.0" armature="0.1" limited="false"/>
        <joint name="z_rotate_{agent_id}" type="hinge" axis="0 0 1" damping="0.5" armature="0.1" limited="false"/>

        <geom name="robot_base_collision_{agent_id}" type="box" size="{base_half_depth} {base_half_width} {base_half_height}"
              pos="0 0 {base_pos_z}" class="collision" rgba="{rgba}" mass="4.9"/>

        <body name="kicker_body_{agent_id}" pos="{kicker_body_pos_x} 0 {kicker_body_pos_z}">
            <joint name="kicker_slide_{agent_id}" type="slide" axis="1 0 0" limited="true" range="0 0.05" damping="5" stiffness="0" armature="0.01"/>

            <geom name="kicker_plate_{agent_id}" type="box" size="{kicker_half_thick} {kicker_half_width} {kicker_half_height}"
                  pos="{kicker_geom_pos_x} 0 0" class="collision" rgba="1 0.5 0 1" mass="0.1" friction="1.0 0.1 0.1"/>
        </body>
    </body>""")

def get_robot_actuators(agent_id: int) -> str:
    """Generates the <actuator> definitions for a single robot."""
    return textwrap.dedent(f"""
        <velocity name="x_vel_{agent_id}" joint="x_slide_{agent_id}" kv="22" ctrlrange="-3 3"/>
        <velocity name="y_vel_{agent_id}" joint="y_slide_{agent_id}" kv="22" ctrlrange="-3 3"/>
        <velocity name="rot_vel_{agent_id}" joint="z_rotate_{agent_id}" kv="5"/>
        <position name="kicker_pos_{agent_id}" joint="kicker_slide_{agent_id}" kp="800" kv="20" ctrlrange="0 1" ctrllimited="true"/>""")

def get_robot_exclude(agent_id: int) -> str:
    """Generates the <exclude> tag for self-collision within a single robot."""
    # Only the exclude tag is truly per-robot specific in the contact section now
    return textwrap.dedent(f"""
        <exclude body1="robot_planar_{agent_id}" body2="kicker_body_{agent_id}"/>""")

# === Construct the Final XML ===

# Generate parts for all agents
all_robot_bodies = "".join([get_robot_body(cfg['id'], cfg['pos'], cfg['rgba']) for cfg in AGENT_CONFIG])
all_robot_actuators = "".join([get_robot_actuators(cfg['id']) for cfg in AGENT_CONFIG])
all_robot_excludes = "".join([get_robot_exclude(cfg['id']) for cfg in AGENT_CONFIG])

# Define the main XML template structure
# Note: Collision pairs are now defined globally here, not generated per-robot
#       We assume agent ID 0 for pairs involving the robot in this single-agent setup.
#       If TOTAL_AGENTS > 1, the pair generation would need a loop or more complex logic.
final_xml_template = textwrap.dedent(f"""\
<mujoco model="robot_soccer_v3"> <option gravity="0 0 -9.81"
          timestep="0.002"
          solver="Newton"          iterations="5"           ls_iterations="10"       jacobian="dense">        <flag eulerdamp="disable"/>      </option>
  <default>
    <geom contype="0" conaffinity="0" group="1"/>
    <default class="visual">
       <geom contype="0" conaffinity="0" group="0"/> </default>
    <default class="collision">
        <geom contype="1" conaffinity="1" group="1" friction="0.7 0.1 0.1" solref="0.02 1" solimp="0.9 0.95 0.001"/> </default>
  </default>
  <asset>
    </asset>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <camera name="top" pos="0 0 15" zaxis="0 0 1"/>

    <geom name="field" type="plane" size="7.5 6 0.5" pos="0 0 0" class="collision" rgba="0.48 0.78 0.27 1"/>

    <body name="left_goal" pos="-6 0 0">
        <site name="left_goal_line" pos="-0.1 0 0.001" size="0.1 0.9 0.001" rgba="1 0 0 0.5" type="box" class="visual"/>
        <geom name="left_goal_side_top" type="box" size="0.1 0.01 0.09" pos="-0.1 0.9 0.09" class="collision" rgba="0.1 0.1 0.8 1"/>
        <geom name="left_goal_side_bottom" type="box" size="0.1 0.01 0.09" pos="-0.1 -0.9 0.09" class="collision" rgba="0.1 0.1 0.8 1"/>
        <geom name="left_goal_backside" type="box" pos="-0.2 0 0.09" size="0.01 0.9 0.09" class="collision" rgba="0.1 0.1 0.8 0.7"/>
    </body>
    <body name="right_goal" pos="6 0 0">
        <site name="right_goal_line" pos="0.1 0 0.001" size="0.1 0.9 0.001" rgba="1 0 0 0.5" type="box" class="visual"/>
        <geom name="right_goal_side_top" type="box" size="0.1 0.01 0.09" pos="0.1 0.9 0.09" class="collision" rgba="0.1 0.1 0.8 1"/>
        <geom name="right_goal_side_bottom" type="box" size="0.1 0.01 0.09" pos="0.1 -0.9 0.09" class="collision" rgba="0.1 0.1 0.8 1"/>
        <geom name="right_goal_backside" type="box" pos="0.2 0 0.09" size="0.01 0.9 0.09" class="collision" rgba="0.1 0.1 0.8 0.7"/>
    </body>

    <site name="field_line_north" pos="0 4.5 0.001" size="6 0.008 0.001" rgba="1 1 1 1" type="box" class="visual"/>
    <site name="field_line_south" pos="0 -4.5 0.001" size="6 0.008 0.001" rgba="1 1 1 1" type="box" class="visual"/>
    <site name="field_line_east" pos="6 0 0.001" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box" class="visual"/>
    <site name="field_line_west" pos="-6 0 0.001" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box" class="visual"/>
    <site name="field_line_center_vert" pos="0 0 0.001" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box" class="visual"/>
    <site name="field_line_center_hor" pos="0 0 0.001" size="6 0.008 0.001" rgba="1 1 1 1" type="box" class="visual"/>

    {textwrap.indent(all_robot_bodies, '    ')}

    <body name="golf_ball" pos="0 0 0.0115">
        <joint name="ball_x_slide" type="slide" axis="1 0 0" limited="false" damping="0.05" armature="0.001"/>
        <joint name="ball_y_slide" type="slide" axis="0 1 0" limited="false" damping="0.05" armature="0.001"/>
        <geom name="golf_ball_geom" type="sphere" size="0.0115" class="collision" rgba="1 0 0 1" mass="0.046" friction="0.4 0.1 0.1" solref="0.01 1" solimp="0.9 0.98 0.001"/>
    </body>
    
  </worldbody>
  <actuator>
{textwrap.indent(all_robot_actuators, '    ')}
  </actuator>
  <contact>
    <pair geom1="robot_base_collision_0" geom2="field"/>
    <pair geom1="kicker_plate_0"       geom2="field"/>
    <pair geom1="golf_ball_geom"         geom2="field"/>

    <pair geom1="kicker_plate_0"       geom2="golf_ball_geom"/>

    <pair geom1="golf_ball_geom" geom2="left_goal_side_top"/>
    <pair geom1="golf_ball_geom" geom2="left_goal_side_bottom"/>
    <pair geom1="golf_ball_geom" geom2="left_goal_backside"/>
    <pair geom1="golf_ball_geom" geom2="right_goal_side_top"/>
    <pair geom1="golf_ball_geom" geom2="right_goal_side_bottom"/>
    <pair geom1="golf_ball_geom" geom2="right_goal_backside"/>

    <pair geom1="robot_base_collision_0" geom2="left_goal_side_top"/>
    <pair geom1="robot_base_collision_0" geom2="left_goal_side_bottom"/>
    <pair geom1="robot_base_collision_0" geom2="left_goal_backside"/>
    <pair geom1="kicker_plate_0"       geom2="left_goal_side_top"/>
    <pair geom1="kicker_plate_0"       geom2="left_goal_side_bottom"/>
    <pair geom1="kicker_plate_0"       geom2="left_goal_backside"/>
    <pair geom1="robot_base_collision_0" geom2="right_goal_side_top"/>
    <pair geom1="robot_base_collision_0" geom2="right_goal_side_bottom"/>
    <pair geom1="robot_base_collision_0" geom2="right_goal_backside"/>
    <pair geom1="kicker_plate_0"       geom2="right_goal_side_top"/>
    <pair geom1="kicker_plate_0"       geom2="right_goal_side_bottom"/>
    <pair geom1="kicker_plate_0"       geom2="right_goal_backside"/>

    {textwrap.indent(all_robot_excludes, '    ')}

  </contact>
  </mujoco>
""")


# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Write the generated XML to the file
with open(output_filename, "w") as f:
    f.write(final_xml_template)

print(f"Generated XML file at: {output_filename}")