import os
import json
import roboverse
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo


# Define training and validation tasks
TRAIN_TASKS = [
    "HumanoidBench-Stand-v0",
    "HumanoidBench-Walk-v0",
    "HumanoidBench-Run-v0",
    "HumanoidBench-Sit-v0",
]

VALIDATION_TASKS = [
    "HumanoidBench-Push-v0",
    "HumanoidBench-Lift-v0",
    "HumanoidBench-OpenDrawer-v0",
    "HumanoidBench-PickPlace-v0",
]

ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def train_model(task, algo="PPO", simulator="mujoco", timesteps=5000):
    """Train a model on one RoboVerse task."""
    env = roboverse.make(task, sim=simulator, gui=False)
    model_cls = ALGORITHMS[algo]
    model = model_cls("MlpPolicy", env, verbose=1)

    print(f"\n=== Training {algo} on {task} for {timesteps} timesteps ===")
    model.learn(total_timesteps=timesteps)

    save_path = f"models/{algo.lower()}_{task}"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    env.close()

    return save_path


def validate_model(
    model_path, 
    task, 
    algo="PPO", 
    simulator="mujoco", 
    episodes=5, 
    record_video=False
):
    """Validate a trained model on a given RoboVerse task."""
    if record_video:
        env = roboverse.make(task, sim=simulator, gui=False)
        env = RecordVideo(env, video_folder="videos", name_prefix=f"{algo}_{task}")
    else:
        env = roboverse.make(task, sim=simulator, gui=False)

    model_cls = ALGORITHMS[algo]
    model = model_cls.load(model_path, env=env)

    print(f"\n=== Validating {algo} on {task} ===")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True
    )
    print(f"Validation reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return {"task": task, "algo": algo, "mean_reward": mean_reward, "std_reward": std_reward}


def run_experiments(
    train_tasks=TRAIN_TASKS,
    val_tasks=VALIDATION_TASKS,
    algos=("PPO", "SAC", "TD3"),
    timesteps=10000,
    eval_episodes=5,
    simulator="mujoco",
    record_video=False,
):
    """Run training on train_tasks and validation on both train+val tasks."""
    results = {"train": [], "validation": []}

    # Train models
    for task in train_tasks:
        for algo in algos:
            model_path = train_model(task, algo=algo, simulator=simulator, timesteps=timesteps)

            # Validate on training task
            res = validate_model(
                model_path, task, algo=algo, simulator=simulator, episodes=eval_episodes
            )
            results["train"].append(res)

    # Cross-validation on unseen tasks
    for task in val_tasks:
        for algo in algos:
            # Reuse first training task model for validation
            model_path = f"models/{algo.lower()}_{train_tasks[0]}"
            res = validate_model(
                model_path,
                task,
                algo=algo,
                simulator=simulator,
                episodes=eval_episodes,
                record_video=record_video,
            )
            results["validation"].append(res)
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Experiment Results Saved to results/experiment_results.json ===")
    return results
