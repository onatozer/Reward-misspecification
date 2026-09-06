import sys
sys.path.append("./safe-grid-gym")
import gym
import safe_grid_gym  # registers envs with old gym registry
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, ProgressBarCallback

import config
from config import *
from lad import LAD


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a specific algorithm.")

    # Add the algorithm string argument
    parser.add_argument(
        "-a", "--algorithm_str",
        type=str,
        default="A2C",
        help="Name of the RL algorithm to run (ex 'PPO', 'A2C')"
    )
    
    parser.add_argument(
        "-n", "--num_steps",
        type=int,
        default=1_000_000,
        help="Number of training steps the algorithm undergoes"
    )

    # Parse arguments
    return parser


str_to_alg = {
    "PPO": PPO,
    "A2C": A2C,
    "LAD": LAD,
}

class TrainingPerformanceCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.timesteps = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.hidden_rewards = []

    def _on_step(self) -> bool:

        for info in self.locals["infos"]:

            if "episode" in info:
                # print("-"*80)
                # print(info)
                self.timesteps.append(self.num_timesteps)
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.hidden_rewards.append(info["episode"]["hidden_reward"])

        return True

    def plot_training_run(self, plotname: str) -> pd.DataFrame:
        df = pd.DataFrame({
            "timesteps": self.timesteps,
            "reward": self.episode_rewards,
            "hidden_reward": self.hidden_rewards,
            "episode_length": self.episode_lengths,
        })

        x = df["timesteps"]
        y = df["reward"]

        plt.plot(x, y)
        plt.savefig(plotname)

        return df

        


def train_eval_loop(env_str: str = ENVIRONMENT_SETTINGS[0], algorithm_str: str = "PPO", num_steps: int = 1_000_000):
    # Dynamically initialize the algorithm based off of which algorithm string and environment setting was passed in
    env = gym.make(env_str)
    # env = make_vec_env(env_str, n_envs=NUM_ENVS)
    env = Monitor(env, info_keywords=("hidden_reward",))

    kwarg_dict = getattr(config, f"{algorithm_str}_KWARGS")
    kwarg_dict["env"] = env

    algorithm = str_to_alg[algorithm_str]
    model = algorithm(**kwarg_dict)
     
    model_str = f"{algorithm_str}_{env_str}"

    train_callback = TrainingPerformanceCallback(verbose=True)

    callbacks = CallbackList([
        EntropyAnnealingCallback(initial=0.05, final=0.01, anneal_steps=int(num_steps/2)),
        CheckpointCallback(save_freq=int(num_steps/5), save_path="model_weights", name_prefix = model_str),
        train_callback,
        ]
    )

    model.learn(total_timesteps=num_steps, progress_bar=True, callback=callbacks)

    df = train_callback.plot_training_run("test.png")
    df.to_csv(f"{model_str}.csv")

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


def eval_model(model_str: str):
    algorithm, env_str, num_steps = model_str.split("_")
    env = gym.make(env_str) 
    model = str_to_alg[algorithm]

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
    parser = create_parser()
    args = parser.parse_args()
    for env_str in ENVIRONMENT_SETTINGS[:1]:
        train_eval_loop(algorithm_str=args.algorithm_str, env_str= env_str, num_steps=args.num_steps)



if __name__ == "__main__":
    main()
