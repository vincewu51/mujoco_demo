from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# MuJoCo low-level API (for forward resets after parameter changes)
import mujoco


# -----------------------------
# Domain Randomization Wrappers
# -----------------------------
@dataclass
class VisualRandCfg:
    color_jitter_strength: float = 0.5  # 0..1 range per RGBA channel
    light_pos_jitter: float = 0.5  # meters
    light_dir_jitter: float = 0.5  # radians-equivalent small vec
    light_ambient_range: Tuple[float, float] = (0.1, 0.7)
    light_diffuse_range: Tuple[float, float] = (0.3, 1.0)


class VisualRandomizationWrapper(gym.Wrapper):
    """Randomize visual appearance at every reset.

    Effective mostly for image-based observations. Included here for completeness.
    """

    def __init__(
        self, env: gym.Env, cfg: VisualRandCfg | None = None, seed: int | None = None
    ):
        super().__init__(env)
        self.cfg = cfg or VisualRandCfg()
        self.np_random = np.random.RandomState(seed)

    def _rand_rgba(self, base_rgba: np.ndarray) -> np.ndarray:
        jitter = (
            (self.np_random.rand(*base_rgba.shape) - 0.5)
            * 2.0
            * self.cfg.color_jitter_strength
        )
        out = np.clip(base_rgba + jitter, 0.0, 1.0)
        # Keep alpha as-is if present and nonzero
        if out.shape[-1] == 4:
            out[..., 3] = np.clip(base_rgba[..., 3], 0.5, 1.0)
        return out

    def randomize_visuals(self):
        m = self.unwrapped.model
        d = self.unwrapped.data
        # Geom colors
        if (
            hasattr(m, "geom_rgba")
            and m.geom_rgba is not None
            and m.geom_rgba.shape[0] > 0
        ):
            # Some envs put 0 in alpha to signal "use texture"; we still jitter RGB
            base = m.geom_rgba.copy()
            jittered = self._rand_rgba(base)
            m.geom_rgba[:] = jittered

        # Lights
        if hasattr(m, "nlight") and m.nlight > 0:
            # Positions
            if hasattr(m, "light_pos"):
                m.light_pos[:] += (
                    self.np_random.randn(m.nlight, 3) * self.cfg.light_pos_jitter
                )
            # Directions
            if hasattr(m, "light_dir"):
                m.light_dir[:] += (
                    self.np_random.randn(m.nlight, 3) * self.cfg.light_dir_jitter
                )
            # Ambient & Diffuse
            if hasattr(m, "light_ambient"):
                low, high = self.cfg.light_ambient_range
                m.light_ambient[:] = self.np_random.uniform(
                    low, high, size=(m.nlight, 3)
                )
            if hasattr(m, "light_diffuse"):
                low, high = self.cfg.light_diffuse_range
                m.light_diffuse[:] = self.np_random.uniform(
                    low, high, size=(m.nlight, 3)
                )

        mujoco.mj_forward(m, d)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self.randomize_visuals()
        except Exception:
            # If the env doesn't expose expected fields, fail gracefully
            pass
        return obs, info


@dataclass
class PhysicsRandCfg:
    friction_range: Tuple[float, float] = (0.5, 1.5)  # multiplicative scale
    damping_range: Tuple[float, float] = (0.5, 1.5)
    mass_range: Tuple[float, float] = (0.8, 1.2)
    actuator_strength_range: Tuple[float, float] = (0.8, 1.2)  # via gear scaling


class PhysicsRandomizationWrapper(gym.Wrapper):
    """Randomize physics parameters each reset for robustness training."""

    def __init__(
        self, env: gym.Env, cfg: PhysicsRandCfg | None = None, seed: int | None = None
    ):
        super().__init__(env)
        self.cfg = cfg or PhysicsRandCfg()
        self.np_random = np.random.RandomState(seed)
        self._ref: Dict[str, np.ndarray] | None = None

    def _snapshot_defaults(self):
        m = self.unwrapped.model
        if self._ref is None:
            self._ref = dict(
                geom_friction=(
                    m.geom_friction.copy() if hasattr(m, "geom_friction") else None
                ),
                dof_damping=m.dof_damping.copy() if hasattr(m, "dof_damping") else None,
                body_mass=m.body_mass.copy() if hasattr(m, "body_mass") else None,
                actuator_gear=(
                    m.actuator_gear.copy() if hasattr(m, "actuator_gear") else None
                ),
            )

    def _apply_scales(
        self, fric_scale: float, damp_scale: float, mass_scale: float, gear_scale: float
    ):
        m = self.unwrapped.model
        d = self.unwrapped.data
        if self._ref is None:
            self._snapshot_defaults()
        ref = self._ref
        # Friction (each geom: [slide, spin, roll])
        if ref.get("geom_friction") is not None and hasattr(m, "geom_friction"):
            m.geom_friction[:] = np.clip(ref["geom_friction"] * fric_scale, 0.0, None)
        # Damping per dof
        if ref.get("dof_damping") is not None and hasattr(m, "dof_damping"):
            m.dof_damping[:] = np.clip(ref["dof_damping"] * damp_scale, 0.0, None)
        # Mass per body
        if ref.get("body_mass") is not None and hasattr(m, "body_mass"):
            m.body_mass[:] = np.clip(ref["body_mass"] * mass_scale, 1e-6, None)
        # Actuator gear scaling as a proxy for strength
        if ref.get("actuator_gear") is not None and hasattr(m, "actuator_gear"):
            m.actuator_gear[:] = ref["actuator_gear"] * gear_scale
        mujoco.mj_forward(m, d)

    def randomize_physics(self):
        rs = self.np_random.uniform
        fric_scale = rs(self.cfg.friction_range[0], self.cfg.friction_range[1])
        damp_scale = rs(self.cfg.damping_range[0], self.cfg.damping_range[1])
        mass_scale = rs(self.cfg.mass_range[0], self.cfg.mass_range[1])
        gear_scale = rs(
            self.cfg.actuator_strength_range[0], self.cfg.actuator_strength_range[1]
        )
        self._apply_scales(fric_scale, damp_scale, mass_scale, gear_scale)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self._snapshot_defaults()
            self.randomize_physics()
        except Exception:
            pass
        return obs, info


# -----------------------------
# Deterministic Perturbation for Evaluation
# -----------------------------
@dataclass
class Perturbation:
    friction_scale: float = 1.0
    damping_scale: float = 1.0
    mass_scale: float = 1.0
    actuator_strength_scale: float = 1.0


class DeterministicPerturbationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, perturb: Perturbation):
        super().__init__(env)
        self.perturb = perturb
        self._ref: Dict[str, np.ndarray] | None = None

    def _snapshot_defaults(self):
        m = self.unwrapped.model
        if self._ref is None:
            self._ref = dict(
                geom_friction=(
                    m.geom_friction.copy() if hasattr(m, "geom_friction") else None
                ),
                dof_damping=m.dof_damping.copy() if hasattr(m, "dof_damping") else None,
                body_mass=m.body_mass.copy() if hasattr(m, "body_mass") else None,
                actuator_gear=(
                    m.actuator_gear.copy() if hasattr(m, "actuator_gear") else None
                ),
            )

    def _apply(self):
        m = self.unwrapped.model
        d = self.unwrapped.data
        ref = self._ref
        if ref is None:
            return
        if ref.get("geom_friction") is not None and hasattr(m, "geom_friction"):
            m.geom_friction[:] = np.clip(
                ref["geom_friction"] * self.perturb.friction_scale, 0.0, None
            )
        if ref.get("dof_damping") is not None and hasattr(m, "dof_damping"):
            m.dof_damping[:] = np.clip(
                ref["dof_damping"] * self.perturb.damping_scale, 0.0, None
            )
        if ref.get("body_mass") is not None and hasattr(m, "body_mass"):
            m.body_mass[:] = np.clip(
                ref["body_mass"] * self.perturb.mass_scale, 1e-6, None
            )
        if ref.get("actuator_gear") is not None and hasattr(m, "actuator_gear"):
            m.actuator_gear[:] = (
                ref["actuator_gear"] * self.perturb.actuator_strength_scale
            )
        mujoco.mj_forward(m, d)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self._snapshot_defaults()
            self._apply()
        except Exception:
            pass
        return obs, info


# -----------------------------
# Utilities
# -----------------------------


def make_env(
    env_id: str, seed: int, train_visual_rand: bool, train_physics_rand: bool
) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    if train_visual_rand:
        env = VisualRandomizationWrapper(env, seed=seed)
    if train_physics_rand:
        env = PhysicsRandomizationWrapper(env, seed=seed)
    return env


def evaluate(
    model: PPO,
    env_id: str,
    perturb: Perturbation,
    episodes: int = 10,
    seed: int | None = None,
) -> float:
    env = gym.make(env_id)
    env = DeterministicPerturbationWrapper(env, perturb)
    if seed is not None:
        env.reset(seed=seed)
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        done, trunc = False, False
        ep_ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


class TrainingLogger(BaseCallback):
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and self.verbose:
            ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
            if ep_rew_mean is not None:
                print(f"[step {self.n_calls}] rollout/ep_rew_mean={ep_rew_mean:.1f}")
        return True


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="Hopper-v4",
        help="Gymnasium MuJoCo env id, e.g. Hopper-v4, HalfCheetah-v4, Ant-v4",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--train-visual-rand", type=int, default=1)
    parser.add_argument("--train-physics-rand", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="./ppo_model")
    parser.add_argument(
        "--load-path",
        type=str,
        default="",
        help="If provided, loads an existing model from this path (zip)",
    )
    parser.add_argument("--eval-only", type=int, default=0)

    args = parser.parse_args()

    # Training env (with domain randomization)
    def _thunk():
        return make_env(
            env_id=args.env_id,
            seed=args.seed,
            train_visual_rand=bool(args.train_visual_rand),
            train_physics_rand=bool(args.train_physics_rand),
        )

    venv = VecMonitor(DummyVecEnv([_thunk]))

    # Create or load PPO
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model from {args.load_path}")
        model = PPO.load(args.load_path, env=venv, print_system_info=True)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            learning_rate=3e-4,
            clip_range=0.2,
            seed=args.seed,
        )

    if not args.eval_only:
        print("\n==> Training with domain randomization...")
        model.learn(total_timesteps=args.total_timesteps, callback=TrainingLogger())
        save_path = (
            args.save_path
            if args.save_path.endswith(".zip")
            else (args.save_path + ".zip")
        )
        model.save(save_path)
        print(f"Model saved to: {save_path}")

    else:
        print("\n==> Skipping training; running evaluation only.")

    # Robustness Evaluation: sweep deterministic perturbation scales
    print("\n==> Robustness evaluation (deterministic sweeps)...")
    sweep_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
    headers = [
        "fric",
        "damp",
        "mass",
        "act",
        f"avg_return@{args.eval_episodes}ep",
    ]
    rows = []

    for fric in sweep_scales:
        for damp in sweep_scales:
            for mass in sweep_scales:
                for act in sweep_scales:
                    ret = evaluate(
                        model,
                        env_id=args.env_id,
                        perturb=Perturbation(
                            friction_scale=fric,
                            damping_scale=damp,
                            mass_scale=mass,
                            actuator_strength_scale=act,
                        ),
                        episodes=args.eval_episodes,
                        seed=args.seed,
                    )
                    rows.append([fric, damp, mass, act, ret])
                    print(
                        f"scales (f={fric:.1f}, d={damp:.1f}, m={mass:.1f}, a={act:.1f}) -> return {ret:.1f}"
                    )

    arr = np.array(rows)
    best_idx = int(np.argmax(arr[:, -1]))
    worst_idx = int(np.argmin(arr[:, -1]))
    best = arr[best_idx]
    worst = arr[worst_idx]

    print("\n==> Summary")
    print("columns:", " | ".join(headers))
    print("best : ", best)
    print("worst: ", worst)
