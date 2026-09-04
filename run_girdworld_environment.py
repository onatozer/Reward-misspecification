import sys
sys.path.append("./safe-grid-gym")
# import gymnasium as gym
import gym
import safe_grid_gym  # registers envs with old gym registry
from stable_baselines3 import PPO, DQN, A2C
import numpy as np

# from ai_safety_gridworlds.helpers import factory
# from ai_safety_gridworlds.environments.shared.safety_game import Actions

ENVIRONMENT_SETTINGS = [
    "DistributionalShift-v0",
    "BoatRace-v0", # Reward-hacking
    "TomatoWatering-v0", # Reward-hacking
    "AbsentSupervisor-v0", 
    "IslandNavigation-v0", # Safe exploration
    "SideEffectsSokoban-v0",
]


def train_eval_loop(env_str: str = ENVIRONMENT_SETTINGS[0]):
    env = gym.make(env_str)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=100_000)

    # Evaluate
    obs = env.reset()
    episode_return = 0
    episode_hidden = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_return += reward
        # CRITICAL: info contains the hidden (safety) reward
        if info.get("hidden_reward") is not None:
            episode_hidden += info["hidden_reward"]
        
        if done:
            print(f"Episode done. Observed return: {episode_return:.1f}, "
                  f"Hidden (safety) return: {episode_hidden:.1f}")
            episode_return = 0
            episode_hidden = 0
            obs = env.reset()

    


def main():
    # env = factory.
    env = gym.make(ENVIRONMENT_SETTINGS[0])
    print("Observation spec:", env.observation_space)
    print("Action spec:", env.action_space)
    # print(Actions)
    # model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    # model.learn(10_000)
    train_eval_loop()



if __name__ == "__main__":
    main()
