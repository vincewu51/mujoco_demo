from stable_baselines3 import PPO
import roboverse

def train_humanoidbench(simulator='mujoco'):
    env = roboverse.make('HumanoidBench-Stand-v0', sim=simulator, gui=False)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_humanoidbench_stand")

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()