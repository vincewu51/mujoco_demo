import mujoco
import mujoco.viewer
import numpy as np
import time

# ----------------
# Load Model
# ----------------
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body>
        <joint name="joint2" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <site name="ee" pos="0.2 0 0" size="0.01" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

# Start and goal
q_start = np.array([0.0, 0.0])
q_goal = np.array([np.pi/2, -np.pi/4])
n_steps = 100
trajectory = np.linspace(q_start, q_goal, n_steps)

# ----------------
# Validation Functions
# ----------------
def check_collision(model, data):
    """Return True if collision detected."""
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):  # number of contacts
        return True
    return False

def end_effector_pos(model, data):
    """Return end-effector position from site 'ee'."""
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site("ee").id].copy()

# ----------------
# Validate trajectory
# ----------------
collisions = []
ee_positions = []

for q in trajectory:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    
    # record EE pos
    ee_positions.append(end_effector_pos(model, data))
    
    # check collision
    if check_collision(model, data):
        collisions.append(q)

print(f"Trajectory validation:")
print(f"- Total steps: {n_steps}")
print(f"- Collisions detected: {len(collisions)}")
print(f"- Start EE: {ee_positions[0]}")
print(f"- Goal EE:  {ee_positions[-1]}")
