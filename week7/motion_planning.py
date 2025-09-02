import mujoco
import mujoco.viewer
import numpy as np
import time

# ----------------
# Load Model (2-link arm + table + object)
# ----------------
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1"/>
    <geom name="table" type="box" size="0.2 0.2 0.05" pos="0.3 0 0.05" rgba="0.6 0.3 0.2 1"/>
    
    <body name="arm" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body>
        <joint name="joint2" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <site name="ee" pos="0.2 0 0" size="0.01" rgba="1 0 0 1"/>
      </body>
    </body>

    <body name="object" pos="0.3 0 0.1">
      <freejoint/>
      <geom name="box" type="box" size="0.02 0.02 0.02" rgba="0.2 0.2 1 1"/>
    </body>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

# ----------------
# Helper functions
# ----------------
def check_collision(model, data):
    """Return True if collision detected."""
    mujoco.mj_forward(model, data)
    return data.ncon > 0

def end_effector_pos(model, data):
    """Return end-effector position from site 'ee'."""
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site("ee").id].copy()

def interpolate_path(q_start, q_goal, n_steps=50):
    return np.linspace(q_start, q_goal, n_steps)

def execute_trajectory(model, data, trajectory, name="Task"):
    ee_positions = []
    collisions = []
    for q in trajectory:
        data.qpos[:len(q)] = q
        mujoco.mj_forward(model, data)
        ee_positions.append(end_effector_pos(model, data))
        if check_collision(model, data):
            collisions.append(q)
    print(f"{name}: steps={len(trajectory)}, collisions={len(collisions)}")
    print(f"  Start EE: {ee_positions[0]}")
    print(f"  End   EE: {ee_positions[-1]}")
    return ee_positions

tasks = [
    ("Reach", np.array([0.0, 0.0]), np.array([np.pi/3, -np.pi/4])),
    ("Move to object", np.array([np.pi/3, -np.pi/4]), np.array([np.pi/2, -np.pi/2])),
    ("Lift object", np.array([np.pi/2, -np.pi/2]), np.array([np.pi/2, -np.pi/3])),
    ("Place on table", np.array([np.pi/2, -np.pi/3]), np.array([np.pi/4, -np.pi/4]))
]

for task_name, q_start, q_goal in tasks:
    traj = interpolate_path(q_start, q_goal, n_steps=80)
    execute_trajectory(model, data, traj, task_name)
