import torch.nn as nn
import gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, ProgressBarCallback


ENVIRONMENT_SETTINGS = [
    "DistributionalShift-v0",
    "BoatRace-v0", # Reward-hacking
    "TomatoWatering-v0", # Reward-hacking
    "AbsentSupervisor-v0", 
    "IslandNavigation-v0", # Safe exploration
    "SideEffectsSokoban-v0",
]

str_to_alg = {
    "PPO": PPO,
    "A2C": A2C
}

# ------------------------------------------------------------------
# 1. LR schedule: anneal linearly from 5e-4 -> 0 over first 900k steps
# ------------------------------------------------------------------
def lr_schedule(progress_remaining: float) -> float:
    """Assumes total_timesteps = 1_000_000."""
    if progress_remaining < 0.1:          # past 900k steps
        return 0.0
    return 5e-4 * (progress_remaining - 0.1) / 0.9


# ------------------------------------------------------------------
# 2. Entropy annealing callback (paper anneals β for some envs)
# ------------------------------------------------------------------
class EntropyAnnealingCallback(BaseCallback):
    """
    For distributional shift the paper anneals β to 0 or 0.01
    over 500_000 timesteps. Adjust initial/final to your env.
    """
    def __init__(self, initial: float = 0.05, final: float = 0.01,
                 anneal_steps: int = 500_000, verbose: int = 0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.anneal_steps)
        self.model.ent_coef = self.initial + (self.final - self.initial) * progress
        return True


# ------------------------------------------------------------------
# 3. Reward normalization wrapper (paper divides by max |reward|)
# ------------------------------------------------------------------
class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, max_abs_reward: float):
        super().__init__(env)
        self.scale = max_abs_reward

    def reward(self, r):
        return r / self.scale


# ------------------------------------------------------------------
# 5. A2C kwargs matching the paper
# ------------------------------------------------------------------
A2C_KWARGS = {
    "policy": "MlpPolicy",          # paper uses MLP, not CNN
    "learning_rate": lr_schedule,   # 5e-4 -> 0 over 900k steps
    "n_steps": 5,                   # "policy unrolled over 5 time steps"
    "gamma": 0.99,                  # discounting
    "gae_lambda": 1.0,              # standard A2C (no GAE)
    "ent_coef": 0.05,               # pick from [0.01, 0.1] based on env
    "vf_coef": 0.25,                # baseline loss weight
    "max_grad_norm": 40,            # gradient clipping by global norm
    "rms_prop_eps": 0.1,            # RMSProp ε
    "use_rms_prop": True,           # use RMSProp (α=0.99 is SB3 default)
    "verbose": 1,
    "policy_kwargs": {
        "net_arch": [100, 100],     # two hidden layers, 100 nodes each
        "activation_fn": nn.ReLU,
    },
}


PPO_KWARGS = {
  "policy": "MlpPolicy",          # paper uses MLP, not CNN
    "learning_rate": lr_schedule,   # 5e-4 -> 0 over 900k steps
    "n_steps": 5,                   # "policy unrolled over 5 time steps"
    "gamma": 0.99,                  # discounting
    "gae_lambda": 1.0,              # standard A2C (no GAE)
    "ent_coef": 0.05,               # pick from [0.01, 0.1] based on env
    "vf_coef": 0.25,                # baseline loss weight
    "max_grad_norm": 40,            # gradient clipping by global norm
    "verbose": 1,
    "policy_kwargs": {
        "net_arch": [100, 100],     # two hidden layers, 100 nodes each
        "activation_fn": nn.ReLU,
    },
    
#     "policy_kwargs": {
#         # Bigger critic than actor — the value function is your bottleneck
#         "net_arch": {"pi": [256, 256], "vf": [512, 512]},
#         "activation_fn": nn.ELU,   # often more stable than ReLU in RL
#         "share_features_extractor": False,  # separate CNN for actor & critic
#         "ortho_init": True,
#     },
}