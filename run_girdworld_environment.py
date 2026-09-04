import sys
sys.path.append("./safe-grid-gym")
import gym
import safe_grid_gym  # registers envs with old gym registry
import numpy as np
import config
from config import *
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, ProgressBarCallback


class TrainingPerformanceCallback(BaseCallback):
    """
    Tracks performance on the actual training environment.

    This records the returns obtained by the trajectories used for training;
    it does NOT run separate evaluation episodes.

    Parameters
    ----------
    log_freq : int
        How often, in training timesteps, to record a performance snapshot.

    window_size : int
        Number of most recent completed episodes used to compute the
        rolling mean/std.

    verbose : int
        SB3 callback verbosity.
    """

    def __init__(
        self,
        log_freq: int = 1_000,
        window_size: int = 100,
    ):
        super().__init__()

        self.log_freq = log_freq
        self.window_size = window_size

        # Raw episode-level data
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_timesteps = []

        # Intermediate performance snapshots
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []

        self._recent_returns = deque(maxlen=window_size)

        self._current_returns = None
        self._current_lengths = None

        self._last_log_timestep = 0

    def _on_training_start(self) -> None:
        # SB3 always internally uses a VecEnv.
        n_envs = self.training_env.num_envs

        self._current_returns = np.zeros(n_envs, dtype=np.float64)
        self._current_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        """
        Called after each training environment step.
        """

        rewards = np.asarray(self.locals["rewards"])
        dones = np.asarray(self.locals["dones"])

        # Add this step's reward to each active episode.
        self._current_returns += rewards
        self._current_lengths += 1

        # Record episodes that terminated on this step.
        for env_idx, done in enumerate(dones):
            if done:
                episode_return = float(self._current_returns[env_idx])
                episode_length = int(self._current_lengths[env_idx])

                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                self.episode_timesteps.append(self.num_timesteps)

                self._recent_returns.append(episode_return)

                # Reset accumulator for that environment.
                self._current_returns[env_idx] = 0.0
                self._current_lengths[env_idx] = 0

        # Save a rolling performance snapshot.
        if (
            self.num_timesteps - self._last_log_timestep
            >= self.log_freq
        ):
            self._record_performance()
            self._last_log_timestep = self.num_timesteps

        return True

    def _record_performance(self) -> None:
        if len(self._recent_returns) == 0:
            return

        recent = np.asarray(self._recent_returns)

        mean_reward = float(np.mean(recent))
        std_reward = float(np.std(recent))

        self.timesteps.append(self.num_timesteps)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)

        # Also make these available to the SB3 logger / TensorBoard.
        self.logger.record(
            "train_performance/mean_reward",
            mean_reward,
        )
        self.logger.record(
            "train_performance/std_reward",
            std_reward,
        )



def train_eval_loop(env_str: str = ENVIRONMENT_SETTINGS[0], algorithm_str: str = "PPO"):
    # Dynamically initialize the algorithm based off of which algorithm string and environment setting was passed in
    env = gym.make(env_str)

    kwarg_dict = getattr(config, f"{algorithm_str}_KWARGS")
    kwarg_dict["env"] = env

    algorithm = str_to_alg[algorithm_str]
    model = algorithm(**kwarg_dict)
     
    num_steps = 500_000
    model_str = f"{algorithm_str}_{env_str}_{num_steps}"

    callbacks = CallbackList([
        EntropyAnnealingCallback(initial=0.05, final=0.01, anneal_steps=500_000),
        CheckpointCallback(save_freq=100_000, save_path="model_weights", name_prefix = model_str),
        TrainingPerformanceCallback(log_freq=1_000, window_size=1_000)
        ]
    )


    # model.load(model_str)
    model.learn(total_timesteps=num_steps, progress_bar=True, callback=callbacks)
    # model.save(os.pathmodel_str)

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
