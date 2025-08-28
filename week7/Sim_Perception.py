"""
Perception + Simulation with MuJoCo

- Adds a camera to the scene
- Captures frames during simulation
- Uses OpenCV to process them
- Saves images as a synthetic dataset
"""

import os
import cv2
import numpy as np
import mujoco
import imageio


# ----------------------------
# Mujoco XML with camera
# ----------------------------
XML = """
<mujoco model="perception_demo">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- Cube -->
    <body name="cube" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0.8 0.3 0.3 1"/>
    </body>

    <!-- Simple robot arm -->
    <body name="arm" pos="0 0 0.1">
      <joint type="hinge" axis="0 0 1" range="-1.57 1.57"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="arm2" pos="0.2 0 0">
        <joint type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>

    <!-- Camera looking at scene -->
    <camera name="fixed_cam" pos="0.5 0.0 0.3" quat="0.7 0.0 0.7 0.0" fovy="60"/>
  </worldbody>
</mujoco>
"""

# ----------------------------
# Sim + Perception loop
# ----------------------------

def run_dataset(n_frames=100, out_dir="dataset"):
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Rendering setup
    width, height = 256, 256
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=2000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    # Use the named camera
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    frames = []
    rect = mujoco.MjrRect(0, 0, width, height)

    for i in range(n_frames):
        # step sim
        mujoco.mj_step(model, data)

        # update scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(rect, scn, con)

        mujoco.mjr_readPixels(rgb, depth, rect, con)
        frame = np.flipud(rgb.copy())  # flip vertical
        frames.append(frame)

        # process with opencv
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # save raw + processed images
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}_edges.png"), edges)

    # save as video too
    imageio.mimsave(os.path.join(out_dir, "simulation.gif"), frames, fps=20)
    print(f"Dataset saved in {out_dir}/")

    # free GPU context
    mujoco.mjr_freeContext(con)


if __name__ == "__main__":
    run_dataset(n_frames=200, out_dir="synthetic_dataset")
