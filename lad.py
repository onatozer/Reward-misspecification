"""Stable-Baselines3 implementation of Learning Advantage Distribution (LAD).

Paper:
    Li & Li, "LAD: Learning Advantage Distribution for Reasoning" (2026)
    https://arxiv.org/abs/2602.20132

This adapts LAD from the paper's contextual-bandit / LLM-response setting to a
standard Gymnasium MDP.  The behavior policy is the policy that generated the
current rollout, and A(s_t, a_t) is estimated with GAE.  For one-step bandit
environments, this reduces to the paper's contextual-bandit setting.

The policy update minimizes the practical LAD objective

    E_{(s,a) ~ pi_old} [ exp(A/eta) * f(
        (pi_theta(a|s) / pi_old(a|s)) / exp(A/eta)
    ) ]

with Jensen-Shannon divergence by default.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv


SelfLAD = TypeVar("SelfLAD", bound="LAD")
Divergence = Literal["js", "hd", "tv", "kl", "rkl", "jf", "ps"]


class LAD(BaseAlgorithm):
    """Learning Advantage Distribution for standard Gymnasium environments.

    The paper formulates LAD for a contextual bandit where a prompt ``x`` is the
    state and a complete response ``y`` is the action.  In a normal MDP, this
    implementation applies the same practical objective per transition:

        x  -> s_t
        y  -> a_t
        A(x, y) -> GAE advantage A_t
        pi_old  -> rollout / behavior policy

    The rollout policy stays fixed while the current rollout is optimized, just
    as required by the LAD likelihood ratio.

    Parameters
    ----------
    policy:
        SB3 actor-critic policy class or one of ``MlpPolicy``, ``CnnPolicy``,
        ``MultiInputPolicy``.
    env:
        Gymnasium environment, VecEnv, or registered environment name.
    learning_rate:
        Optimizer learning rate or SB3 schedule.
    n_steps:
        Number of environment steps collected per environment before an update.
    batch_size:
        Minibatch size used to optimize one rollout.
    n_epochs:
        Number of passes over each rollout.  The LAD reference implementation
        effectively uses PPO-style minibatch optimization; 1 is the most
        conservative on-policy setting.
    gamma:
        Discount factor.
    gae_lambda:
        GAE lambda.
    eta:
        LAD temperature from the paper.  The reference code calls ``1 / eta``
        ``kappa``.  The paper's main math setting uses ``1 / eta = 4``, so the
        default here is ``eta=0.25``.
    divergence:
        f-divergence.  ``"js"`` is the paper's main choice.  ``"hd"`` and
        ``"tv"`` are also strict divergences recommended by the paper.
    normalize_advantage:
        Whiten GAE advantages once over the whole rollout.  This is the closest
        generic-MDP analogue of the normalized advantages used by GRPO while
        keeping the LAD target fixed across optimization epochs.
    vf_coef:
        Value-function loss coefficient.  This critic is only used to estimate
        advantages in sequential environments.
    ent_coef:
        Optional entropy bonus.  LAD does not require auxiliary entropy
        regularization; therefore the default is zero.
    max_grad_norm:
        Gradient clipping norm.
    target_kl:
        Optional early-stop threshold based on approximate KL from the behavior
        policy.  LAD itself does not require PPO clipping.
    numerical_clip:
        Optional clamp for ``log(pi/pi_old)`` and ``A/eta`` before exponentials.
        Set to ``None`` for the literal objective.  A finite value is useful in
        generic Gym tasks whose advantage scale is much less controlled than in
        the paper.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eta: float = 0.25,
        divergence: Divergence = "js",
        normalize_advantage: bool = True,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        target_kl: float | None = None,
        numerical_clip: float | None = 20.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0, got {n_epochs}")
        if divergence not in {"js", "hd", "tv", "kl", "rkl", "jf", "ps"}:
            raise ValueError(f"Unknown LAD divergence: {divergence}")
        if numerical_clip is not None and numerical_clip <= 0:
            raise ValueError("numerical_clip must be positive or None")

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            monitor_wrapper=True,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eta = eta
        self.divergence = divergence
        self.normalize_advantage = normalize_advantage
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.numerical_clip = numerical_clip
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if self.env is not None:
            buffer_size = self.n_steps * self.env.num_envs
            if self.normalize_advantage and buffer_size <= 1:
                raise ValueError(
                    "Advantage normalization needs n_steps * n_envs > 1; "
                    f"got {self.n_steps} * {self.env.num_envs}."
                )

        if self.normalize_advantage and self.batch_size == 1:
            # This is not mathematically invalid because normalization happens
            # before minibatching, but it is almost always an accidental setup.
            if self.verbose >= 1:
                print(
                    "Warning: batch_size=1 is allowed because LAD normalizes the "
                    "whole rollout, but larger minibatches are usually preferable."
                )

        if _init_setup_model:
            self._setup_model()

    @property
    def inv_eta(self) -> float:
        """Return 1 / eta (called ``kappa`` by the authors' released code)."""
        return 1.0 / self.eta

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = self.rollout_buffer_class
        if buffer_cls is None:
            buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

    # ------------------------------------------------------------------
    # LAD objective
    # ------------------------------------------------------------------
    def _bounded_exp(self, x: th.Tensor) -> th.Tensor:
        if self.numerical_clip is not None:
            x = th.clamp(x, -self.numerical_clip, self.numerical_clip)
        return th.exp(x)

    def _lad_loss_per_sample(
        self,
        log_prob: th.Tensor,
        old_log_prob: th.Tensor,
        advantages: th.Tensor,
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        """Compute the practical LAD loss for each sampled transition.

        Let
            l = log pi_theta(a|s) - log pi_old(a|s)
            u = A(s,a) / eta
            x = exp(l - u)

        Then the practical objective is exp(u) * f(x).

        The formulas below follow the authors' released implementation.  JS is
        also exactly Eq. (8) with the JS generator used in the paper.
        """
        log_ratio = log_prob - old_log_prob
        scaled_adv = advantages / self.eta

        if self.numerical_clip is not None:
            safe_log_ratio = th.clamp(log_ratio, -self.numerical_clip, self.numerical_clip)
            safe_scaled_adv = th.clamp(scaled_adv, -self.numerical_clip, self.numerical_clip)
        else:
            safe_log_ratio = log_ratio
            safe_scaled_adv = scaled_adv

        exp_adv = th.exp(safe_scaled_adv)
        log_x = safe_log_ratio - safe_scaled_adv
        x = th.exp(log_x)

        if self.divergence == "js":
            # f(x) = 1/2 [x log x - (x+1) log((x+1)/2)]
            # log((x+1)/2) is written with logaddexp for stability.
            log_mid = th.logaddexp(log_x, th.zeros_like(log_x)) - math.log(2.0)
            f_x = 0.5 * (x * log_x - (x + 1.0) * log_mid)
            lad_loss = exp_adv * f_x

        elif self.divergence == "hd":
            # Hellinger: f(x) = 1/2 (sqrt(x)-1)^2.
            # The released LAD code omits the constant 1/2; retaining it only
            # changes the global loss scale, not the optimum.  We use the paper.
            sqrt_x = th.exp(0.5 * log_x)
            lad_loss = 0.5 * exp_adv * (sqrt_x - 1.0).square()

        elif self.divergence == "tv":
            # Total variation generator used in the paper/code: |x - 1|.
            lad_loss = exp_adv * th.abs(x - 1.0)

        elif self.divergence == "kl":
            # Forward-KL form, equivalent to exp(u) * x log x = ratio * log x.
            ratio = th.exp(safe_log_ratio)
            lad_loss = ratio * log_x

        elif self.divergence == "rkl":
            # Reverse-KL form from the released implementation, up to constants
            # that do not affect the gradient w.r.t. the current policy.
            lad_loss = -exp_adv * safe_log_ratio

        elif self.divergence == "jf":
            # Jeffreys divergence: (x - 1) log x.
            lad_loss = exp_adv * (x - 1.0) * log_x

        elif self.divergence == "ps":
            # Pearson-style divergence used by the released implementation.
            lad_loss = exp_adv * (x - 1.0).square() / x.clamp_min(1e-12)

        else:  # defensive; constructor already validates this
            raise RuntimeError(f"Unsupported divergence {self.divergence}")

        metrics = {
            "log_ratio": log_ratio.detach(),
            "ratio": th.exp(th.clamp(log_ratio.detach(), -20.0, 20.0)),
            "scaled_adv": scaled_adv.detach(),
            "target_ratio": th.exp(th.clamp(scaled_adv.detach(), -20.0, 20.0)),
            "x": x.detach(),
        }
        return lad_loss, metrics

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect data under a fixed behavior policy ``pi_old``."""
        if self._last_obs is None:
            raise RuntimeError("No previous observation. Was _setup_learn() called?")

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        n_steps = 0

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()
            env_actions = actions_np

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    env_actions = self.policy.unscale_action(env_actions)
                else:
                    env_actions = np.clip(env_actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(env_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            actions_for_buffer = actions_np
            if isinstance(self.action_space, spaces.Discrete):
                actions_for_buffer = actions_for_buffer.reshape(-1, 1)

            # Bootstrap TimeLimit truncations, matching SB3's on-policy behavior.
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_for_buffer,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

        # Freeze the advantage target for all optimization epochs.  Standard PPO
        # normalizes each minibatch; doing it once here is a cleaner LAD target.
        if self.normalize_advantage:
            advantages = rollout_buffer.advantages
            mean = float(np.mean(advantages))
            std = float(np.std(advantages))
            rollout_buffer.advantages = (advantages - mean) / (std + 1e-8)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        lad_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        abs_log_ratios: list[float] = []
        mean_target_ratios: list[float] = []
        mean_xs: list[float] = []

        continue_training = True
        last_total_loss: th.Tensor | None = None

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(int(rollout_data.actions.shape[0]))

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages

                per_sample_lad, lad_metrics = self._lad_loss_per_sample(
                    log_prob=log_prob,
                    old_log_prob=rollout_data.old_log_prob,
                    advantages=advantages,
                )
                policy_loss = per_sample_lad.mean()

                # Sequential Gym tasks need a baseline to estimate A_t.  In a
                # one-step bandit, this is simply r - V(s).
                value_loss = F.mse_loss(values, rollout_data.returns)

                if entropy is None:
                    entropy_loss = log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                last_total_loss = total_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio) - 1.0) - log_ratio)

                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping LAD update because approximate KL "
                            f"{approx_kl.item():.4f} exceeded {1.5 * self.target_kl:.4f}."
                        )
                    break

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                lad_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropy_losses.append(float(entropy_loss.detach().cpu()))
                approx_kls.append(float(approx_kl.detach().cpu()))
                abs_log_ratios.append(float(lad_metrics["log_ratio"].abs().mean().cpu()))
                mean_target_ratios.append(float(lad_metrics["target_ratio"].mean().cpu()))
                mean_xs.append(float(lad_metrics["x"].mean().cpu()))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/lad_policy_loss", np.mean(lad_losses) if lad_losses else np.nan)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else np.nan)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else np.nan)
        self.logger.record("train/approx_kl", np.mean(approx_kls) if approx_kls else np.nan)
        self.logger.record("train/abs_log_ratio", np.mean(abs_log_ratios) if abs_log_ratios else np.nan)
        self.logger.record(
            "train/mean_target_ratio", np.mean(mean_target_ratios) if mean_target_ratios else np.nan
        )
        self.logger.record("train/mean_lad_x", np.mean(mean_xs) if mean_xs else np.nan)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/eta", self.eta)
        self.logger.record("train/inv_eta", self.inv_eta)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        if last_total_loss is not None:
            self.logger.record("train/loss", float(last_total_loss.detach().cpu()))

    # ------------------------------------------------------------------
    # SB3 training / logging API
    # ------------------------------------------------------------------
    def dump_logs(self, iteration: int = 0) -> None:
        """Write logs using the same behavior as SB3 OnPolicyAlgorithm."""
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfLAD,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "LAD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfLAD:
        """
        Train LAD with A2C's SB3 logging cadence.

        ``log_interval`` is measured in rollout/update iterations, exactly as for
        SB3 on-policy algorithms. Therefore terminal logs are emitted every
        ``log_interval * n_steps * n_envs`` environment timesteps. The A2C
        default is 100 iterations.
        """
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        callback.on_training_start(locals(), globals())
        if self.env is None:
            raise RuntimeError("LAD.learn() requires an environment")

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
            )
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "policy.optimizer"], []


# Optional convenience aliases matching SB3 algorithm modules.
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
