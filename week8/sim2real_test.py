from __future__ import annotations
import argparse
import os
import csv
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List


import numpy as np
import gymnasium as gym
import mujoco


# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


# Optional plotting and image tools
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
try:
print("Done.")