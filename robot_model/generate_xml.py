import textwrap

# --- Configuration ---
NUM_AGENTS_TEAM_A = 2 # Example: 1 agent in team A
NUM_AGENTS_TEAM_B = 0 # Example: 1 agent in team B
TOTAL_AGENTS = NUM_AGENTS_TEAM_A + NUM_AGENTS_TEAM_B

# Define starting positions and colors (customize as needed)
# Example: Team A blue, Team B green
AGENT_CONFIG = [
    # Team A
    {'id': 0, 'pos': "-2 0 0", 'rgba': "0 0 1 1"},
    {'id': 1, 'pos': "-1 0 0", 'rgba': "0 0 1 1"},
    # Add more team A agents if NUM_AGENTS_TEAM_A > 1
    # {'id': 1, 'pos': "-1 -1 0", 'rgba': "0 0.2 1 1"},

    # Team B
    # {'id': 1, 'pos': "2 0 0", 'rgba': "0 1 0 1"},
    # Add more team B agents if NUM_AGENTS_TEAM_B > 1
    # {'id': 3, 'pos': "1 1 0", 'rgba': "0.2 1 0 1"},
]

# Ensure AGENT_CONFIG matches TOTAL_AGENTS
if len(AGENT_CONFIG) != TOTAL_AGENTS:
    raise ValueError(f"AGENT_CONFIG length ({len(AGENT_CONFIG)}) does not match TOTAL_AGENTS ({TOTAL_AGENTS})")

# === Helper Functions to Generate XML Parts ===

def get_robot_body(agent_id: int, pos: str, rgba: str) -> str:
    """Generates the XML <worldbody> definition for a single robot."""
    return textwrap.dedent(f"""
    <body name="robot_planar_{agent_id}" pos="{pos}">
        <joint name="x_slide_{agent_id}" type="slide" axis="1 0 0" pos="0 0 0" damping="0.8" armature="0.6"/>
        <joint name="y_slide_{agent_id}" type="slide" axis="0 1 0" pos="0 0 0" damping="0.8" armature="0.6"/>
        <joint name="z_rotate_{agent_id}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0" armature="0.5"/>

        <geom name="robot_mesh_{agent_id}" type="mesh" mesh="robot_stl" rgba="{rgba}" mass="5"/>

        <body name="kicker_body_{agent_id}" pos="0 0.065 0.02">
            <joint name="kicker_extend_{agent_id}" type="slide" axis="0 1 0" limited="true" range="0 0.1"
                  damping="0" stiffness="0" armature="0"/>

            <geom name="kicker_plate_geom_{agent_id}" type="box" size="0.055 0.004 0.025" rgba="1 0.5 0 1"
                  mass="0.5" friction="1.8 0.005 0.05" solref="0.01 1" solimp="0.95 0.98 0.001"/>
        </body>
    </body>
    """)

def get_robot_actuators(agent_id: int) -> str:
    """Generates the <actuator> definitions for a single robot."""
    return textwrap.dedent(f"""
        <velocity name="x_vel_{agent_id}" joint="x_slide_{agent_id}" kv="22" ctrlrange="-3 3"/>
        <velocity name="y_vel_{agent_id}" joint="y_slide_{agent_id}" kv="22" ctrlrange="-3 3"/>
        <velocity name="rot_vel_{agent_id}" joint="z_rotate_{agent_id}" kv="5"/>
        <position name="kicker_pos_{agent_id}" joint="kicker_extend_{agent_id}" kp="800" kv="20" ctrlrange="0 1" ctrllimited="true"/>
    """)

def get_robot_contacts(agent_id: int) -> str:
    """Generates the <contact> definitions for a single robot."""
    # Note: Ball geom name "golf_ball_geom" is assumed to be global/unique
    return textwrap.dedent(f"""
        <pair geom1="robot_mesh_{agent_id}" geom2="golf_ball_geom"/>
        <pair geom1="kicker_plate_geom_{agent_id}" geom2="golf_ball_geom"/>
        <exclude body1="robot_planar_{agent_id}" body2="kicker_body_{agent_id}"/>
    """)

# === Construct the Final XML ===

# Generate parts for all agents
all_robot_bodies = "".join([get_robot_body(cfg['id'], cfg['pos'], cfg['rgba']) for cfg in AGENT_CONFIG])
all_robot_actuators = "".join([get_robot_actuators(cfg['id']) for cfg in AGENT_CONFIG])
all_robot_contacts = "".join([get_robot_contacts(cfg['id']) for cfg in AGENT_CONFIG])

# Assemble the full XML string
final_xml = textwrap.dedent(f"""
<mujoco>
  <asset>
    <mesh name="robot_stl" file="robot_decimated.stl" scale="1 1 1"/>
  </asset>

  <worldbody>
    <geom name="field" type="plane" size="7.5 6 0.01" pos="0 0 -0.01" rgba="0.48 0.78 0.27 1"/>

    <body name="left_goal" pos="-6 0 0">
        <site name="left_goal_box" pos="-0.1 0 0" size="0.1 0.9 0.001" rgba="1 0 0 1" type="box"/>
        <geom name="left_goal_side_top" type="box" size="0.1 0.01 0.09" pos="-0.1 0.9 0.09" rgba="0 0 1 1"/>
        <geom name="left_goal_side_bottom" type="box" size="0.1 0.01 0.09" pos="-0.1 -0.9 0.09" rgba="0 0 1 1"/>
        <geom name="left_goal_backside" type="box" pos="-0.2 0 0.09" size="0.01 0.9 0.09" rgba="0 0 1 0.5"/>
    </body>

    <body name="right_goal" pos="6 0 0">
        <site name="right_goal_box" pos="0.1 0 0" size="0.1 0.9 0.001" rgba="1 0 0 1" type="box"/>
        <geom name="right_goal_side_top" type="box" size="0.1 0.01 0.09" pos="0.1 0.9 0.09" rgba="0 0 1 1"/>
        <geom name="right_goal_side_bottom" type="box" size="0.1 0.01 0.09" pos="0.1 -0.9 0.09" rgba="0 0 1 1"/>
        <geom name="right_goal_backside" type="box" pos="0.2 0 0.09" size="0.01 0.9 0.09" rgba="0 0 1 0.5"/>
    </body>

    <site name="field_line_north" pos="0 4.504 0" size="6 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="field_line_south" pos="0 -4.504 0" size="6 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="field_line_east" pos="6.008 0 0" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box"/>
    <site name="field_line_west" pos="-6.008 0 0" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box"/>
    <site name="field_line_vertical" pos="0 0 0" size="0.008 4.5 0.001" rgba="1 1 1 1" type="box"/>
    <site name="field_line_horizontal" pos="0 0 0" size="6 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="left_goal_box_top" pos="-5.1 1.8 0.0005" size="0.9 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="left_goal_box_bottom" pos="-5.1 -1.8 0.0005" size="0.9 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="left_goal_box_front" pos="-4.2 0 0.0005" size="0.008 1.8 0.001" rgba="1 1 1 1" type="box"/>
    <site name="right_goal_box_top" pos="5.1 1.8 0.0005" size="0.9 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="right_goal_box_bottom" pos="5.1 -1.8 0.0005" size="0.9 0.008 0.001" rgba="1 1 1 1" type="box"/>
    <site name="right_goal_box_front" pos="4.2 0 0.0005" size="0.008 1.8 0.001" rgba="1 1 1 1" type="box"/>

    {textwrap.indent(all_robot_bodies, '    ')}
    <body name="golf_ball" pos="0 0 0.0215"> <joint name= "ball_joint" type="free"/>
        <geom name="golf_ball_geom" type="sphere" size="0.0215" rgba="1 1 1 1" mass="0.046" friction="2.408 0.01 0.0035" condim="6"/>
    </body>

  </worldbody>

  <actuator>
    {textwrap.indent(all_robot_actuators, '    ')}
    </actuator>

  <contact>
    {textwrap.indent(all_robot_contacts, '    ')}
    </contact>

</mujoco>
""")

# You can now save final_xml to a file or use it directly
# print(final_xml)
with open("robot_model/multi_robot_soccer_generated.xml", "w") as f:
    f.write(final_xml)

