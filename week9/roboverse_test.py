from stable_baselines3 import PPO, SAC
import roboverse

TASKS = [
    "HumanoidBench-Stand-v0",
    "HumanoidBench-Walk-v0",
    "HumanoidBench-Run-v0",
    "HumanoidBench-Sit-v0",
    "HumanoidBench-Push-v0",
]

def train_and_eval(simulator="mujoco", algo="PPO", timesteps=5000):
    for task in TASKS:
        print(f"\n=== Training {task} with {algo} ===")
        env = roboverse.make(task, sim=simulator, gui=False)

        if algo == "PPO":
            model = PPO("MlpPolicy", env, verbose=1)
        elif algo == "SAC":
            model = SAC("MlpPolicy", env, verbose=1)
        else:
            raise ValueError("Unsupported algo")

        # Train
        model.learn(total_timesteps=timesteps)
        model.save(f"{algo.lower()}_{task}")

        # Evaluate
        obs = env.reset()
        total_reward = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                obs = env.reset()
        print(f"Evaluation reward for {task}: {total_reward:.2f}")

        env.close()
