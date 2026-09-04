import torch.nn as nn

PPO_KWARGS = ppo_kwargs = {
    "policy": "MlpPolicy",
    "verbose": 1,
    
    # Optimizer
    "learning_rate": 3e-4,        # back to default; VecNormalize handles scale
    
    # Rollout & update
    "n_steps": 2048,              # per env → total buffer = 2048 * 8 = 16_384
    "batch_size": 512,            # larger batches, stabler gradients
    "n_epochs": 3,                # fewer epochs to avoid overfitting to stale data
    
    # GAE
    "gamma": 0.99,
    "gae_lambda": 0.95,
    
    # Clipping
    "clip_range": 0.2,
    "clip_range_vf": 0.2,         # stabilize value updates
    
    # Exploration — this is the key to breaking the "stall" policy
    "ent_coef": 0.05,             # was 0.01; force much more risk-taking
    # If your action space is CONTINUOUS, also add:
    # "use_sde": True,
    # "sde_sample_freq": 64,
    
    # Value function
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    
    # Network architecture
    "policy_kwargs": {
        # Bigger critic than actor — the value function is your bottleneck
        "net_arch": {"pi": [256, 256], "vf": [512, 512]},
        "activation_fn": nn.ELU,   # often more stable than ReLU in RL
        "share_features_extractor": False,  # separate CNN for actor & critic
        "ortho_init": True,
    },
}