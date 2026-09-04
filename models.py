from stable_baselines3 import PPO


class PerfectPPO(PPO):
    def __init__(self, args):
        super().__init__(args)
        