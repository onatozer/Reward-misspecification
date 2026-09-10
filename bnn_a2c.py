from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gym import spaces

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class IBNNA2C(A2C):
    """A2C with a BoTorch infinite-width Bayesian neural-network critic.

    The actor remains the ordinary Stable-Baselines3 actor.  The standard SB3
    neural value branch is not used for value prediction or training.  Instead,
    V(s) is represented by a BoTorch ``SingleTaskGP`` whose covariance function
    is ``InfiniteWidthBNNKernel``.

    Policy-gradient weighting is

        A_scaled = E[A] / (2 * confidence_std_mult * Std[A] + uncertainty_eps)

    where Std[A] is estimated by sampling the *joint* I-BNN/GP posterior over
    all value estimates in the rollout and running GAE on each sampled value
    function.  This propagates covariance between V(s_t) and V(s_{t+1}) into
    the advantage uncertainty.

    Notes
    -----
    * This implementation currently supports non-dict ``spaces.Box``
      observations (e.g. the flattened/board observations used by MlpPolicy).
    * Exact GP inference scales cubically in the number of retained critic
      training points, so ``ibnn_max_points`` intentionally bounds the critic
      dataset.
    * ``vf_coef`` is retained for A2C API compatibility but is not used, because
      the critic is fit separately through exact GP inference / marginal
      likelihood rather than the policy optimizer.
    """

    def __init__(
        self,
        *args: Any,
        ibnn_depth: int = 3,
        ibnn_weight_var: float = 10.0,
        ibnn_bias_var: float = 5.0,
        ibnn_noise_var: float = 1e-2,
        ibnn_max_points: int = 512,
        ibnn_optimize_hypers: bool = False,
        ibnn_hyper_fit_every: int = 10,
        n_advantage_samples: int = 32,
        confidence_std_mult: float = 1.0,
        uncertainty_eps: float = 1e-3,
        max_scaled_advantage: float | None = 10.0,
        gp_jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(self.observation_space, spaces.Box):
            raise NotImplementedError(
                "IBNNA2C currently supports only Box observations. "
                "Add a deterministic vectorization step before the I-BNN for Dict observations."
            )

        if n_advantage_samples < 2:
            raise ValueError("n_advantage_samples must be >= 2")
        if confidence_std_mult <= 0:
            raise ValueError("confidence_std_mult must be > 0")
        if ibnn_noise_var <= 0:
            raise ValueError("ibnn_noise_var must be > 0")
        if ibnn_max_points <= 0:
            raise ValueError("ibnn_max_points must be > 0")

        self.ibnn_depth = ibnn_depth
        self.ibnn_weight_var = ibnn_weight_var
        self.ibnn_bias_var = ibnn_bias_var
        self.ibnn_noise_var = ibnn_noise_var
        self.ibnn_max_points = ibnn_max_points
        self.ibnn_optimize_hypers = ibnn_optimize_hypers
        self.ibnn_hyper_fit_every = max(1, ibnn_hyper_fit_every)

        self.n_advantage_samples = n_advantage_samples
        self.confidence_std_mult = confidence_std_mult
        self.uncertainty_eps = uncertainty_eps
        self.max_scaled_advantage = max_scaled_advantage
        self.gp_jitter = gp_jitter

        self._ibnn_model: SingleTaskGP | None = None
        self._gp_train_x: th.Tensor | None = None
        self._gp_train_y: th.Tensor | None = None
        self._ibnn_fit_count = 0

        # The SB3 value branch still exists structurally inside ActorCriticPolicy,
        # but this algorithm never calls it. Freeze it so it cannot accidentally
        # receive updates if the policy implementation changes.
        if hasattr(self.policy, "value_net"):
            for parameter in self.policy.value_net.parameters():
                parameter.requires_grad_(False)
        if hasattr(self.policy.mlp_extractor, "value_net"):
            for parameter in self.policy.mlp_extractor.value_net.parameters():
                parameter.requires_grad_(False)

    # ---------------------------------------------------------------------
    # I-BNN / GP helpers
    # ---------------------------------------------------------------------

    def _make_ibnn_kernel(self) -> InfiniteWidthBNNKernel:
        kernel = InfiniteWidthBNNKernel(
            depth=self.ibnn_depth,
            device=self.device,
        ).to(device=self.device, dtype=th.double)
        kernel.weight_var = self.ibnn_weight_var
        kernel.bias_var = self.ibnn_bias_var
        return kernel

    def _obs_tensor_to_gp_x(self, obs: th.Tensor) -> th.Tensor:
        """Convert a batch of SB3 observations to I-BNN inputs [N, D]."""
        return obs.detach().reshape(obs.shape[0], -1).to(device=self.device, dtype=th.double)

    def _obs_numpy_to_gp_x(self, obs: np.ndarray) -> th.Tensor:
        obs_t = th.as_tensor(obs, device=self.device)
        return self._obs_tensor_to_gp_x(obs_t)

    def _ibnn_value_mean(self, obs: th.Tensor) -> th.Tensor:
        """Posterior mean E[V(s)]. Before the first fit, use the zero GP prior mean."""
        x = self._obs_tensor_to_gp_x(obs)

        if self._ibnn_model is None:
            return th.zeros(x.shape[0], device=self.device, dtype=th.float32)

        self._ibnn_model.eval()
        with th.no_grad():
            posterior = self._ibnn_model.posterior(x, observation_noise=False)
            return posterior.mean.squeeze(-1).to(dtype=th.float32)

    def _sample_value_functions(self, x: th.Tensor, n_samples: int) -> th.Tensor:
        """Draw joint latent-value samples with shape [M, N].

        ``observation_noise=False`` is deliberate: the policy should be scaled by
        epistemic uncertainty in the value function, not by the GP likelihood
        noise used to account for noisy TD/lambda-return targets.
        """
        with th.no_grad():
            if self._ibnn_model is not None:
                self._ibnn_model.eval()
                posterior = self._ibnn_model.posterior(x, observation_noise=False)
                return posterior.rsample(th.Size([n_samples])).squeeze(-1)

            # No critic data yet: sample from the zero-mean I-BNN GP prior.
            kernel = self._make_ibnn_kernel()
            covariance = kernel(x).to_dense()
            covariance = covariance + self.gp_jitter * th.eye(
                x.shape[0], device=x.device, dtype=x.dtype
            )
            prior = th.distributions.MultivariateNormal(
                loc=th.zeros(x.shape[0], device=x.device, dtype=x.dtype),
                covariance_matrix=covariance,
            )
            return prior.rsample(th.Size([n_samples]))

    def _advantage_std_from_current_critic(self) -> th.Tensor:
        """Estimate Std[A_t] for the current rollout via joint posterior samples.

        Returns a flat tensor in the same env-major order produced by
        ``RolloutBuffer.swap_and_flatten``.
        """
        assert self._last_obs is not None
        buffer = self.rollout_buffer
        assert isinstance(buffer, RolloutBuffer)
        assert not buffer.generator_ready, "Call this before rollout_buffer.get()."

        t_steps = buffer.buffer_size
        n_envs = buffer.n_envs

        # [T + 1, n_envs, *obs_shape]
        rollout_plus_last = np.concatenate(
            [buffer.observations, np.asarray(self._last_obs)[None, ...]], axis=0
        )
        x = self._obs_numpy_to_gp_x(
            rollout_plus_last.reshape((t_steps + 1) * n_envs, *buffer.observations.shape[2:])
        )

        # [M, T + 1, n_envs]
        value_samples = self._sample_value_functions(x, self.n_advantage_samples)
        value_samples = value_samples.reshape(self.n_advantage_samples, t_steps + 1, n_envs)

        rewards = th.as_tensor(buffer.rewards, device=self.device, dtype=th.double)
        episode_starts = th.as_tensor(
            buffer.episode_starts, device=self.device, dtype=th.double
        )
        final_dones = th.as_tensor(
            self._last_episode_starts, device=self.device, dtype=th.double
        )

        advantage_samples = th.zeros(
            self.n_advantage_samples,
            t_steps,
            n_envs,
            device=self.device,
            dtype=th.double,
        )
        last_gae = th.zeros(
            self.n_advantage_samples, n_envs, device=self.device, dtype=th.double
        )

        for step in reversed(range(t_steps)):
            if step == t_steps - 1:
                next_non_terminal = 1.0 - final_dones
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]

            current_values = value_samples[:, step, :]
            next_values = value_samples[:, step + 1, :]

            delta = (
                rewards[step].unsqueeze(0)
                + self.gamma * next_values * next_non_terminal.unsqueeze(0)
                - current_values
            )
            last_gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal.unsqueeze(0)
                * last_gae
            )
            advantage_samples[:, step, :] = last_gae

        # Std over posterior function draws: [T, n_envs].
        advantage_std = advantage_samples.std(dim=0, unbiased=False)

        # Match SB3's env-major flattening: [T, n_env] -> [n_env, T] -> [N].
        advantage_std = advantage_std.transpose(0, 1).reshape(-1)
        return advantage_std.to(dtype=th.float32)

    def _fit_ibnn_critic(self, x_new: th.Tensor, y_new: th.Tensor) -> None:
        """Condition the exact I-BNN GP on recent (state, lambda-return) targets."""
        x_new = x_new.detach().to(device=self.device, dtype=th.double)
        y_new = y_new.detach().reshape(-1, 1).to(device=self.device, dtype=th.double)

        if self._gp_train_x is None:
            self._gp_train_x = x_new
            self._gp_train_y = y_new
        else:
            assert self._gp_train_y is not None
            self._gp_train_x = th.cat([self._gp_train_x, x_new], dim=0)
            self._gp_train_y = th.cat([self._gp_train_y, y_new], dim=0)

        # Exact GP complexity is O(N^3), and old on-policy value targets become
        # stale as the actor changes. Keep only the most recent points.
        if self._gp_train_x.shape[0] > self.ibnn_max_points:
            self._gp_train_x = self._gp_train_x[-self.ibnn_max_points :]
            self._gp_train_y = self._gp_train_y[-self.ibnn_max_points :]

        kernel = self._make_ibnn_kernel()
        train_yvar = th.full_like(self._gp_train_y, self.ibnn_noise_var)

        model = SingleTaskGP(
            train_X=self._gp_train_x,
            train_Y=self._gp_train_y,
            train_Yvar=train_yvar,
            covar_module=kernel,
        ).to(device=self.device, dtype=th.double)

        self._ibnn_fit_count += 1
        if self.ibnn_optimize_hypers and (
            self._ibnn_fit_count % self.ibnn_hyper_fit_every == 0
        ):
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Warm-start the next rebuilt GP with the learned I-BNN hyperparameters.
            self.ibnn_weight_var = float(model.covar_module.weight_var.detach().cpu())
            self.ibnn_bias_var = float(model.covar_module.bias_var.detach().cpu())

        model.eval()
        self._ibnn_model = model

    # ---------------------------------------------------------------------
    # Rollout collection: copied from SB3 OnPolicyAlgorithm with one material
    # change: values come from the I-BNN posterior mean, not policy.value_net.
    # ---------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # Actor: ordinary SB3 policy.
                distribution = self.policy.get_distribution(obs_tensor)
                actions = distribution.get_actions(deterministic=False)
                log_probs = distribution.log_prob(actions)
                actions = actions.reshape((-1, *self.action_space.shape))

                # Critic: BoTorch I-BNN posterior mean.
                values = self._ibnn_value_mean(obs_tensor)

            actions_np = actions.cpu().numpy()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions_np, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # Same TimeLimit handling as SB3, but bootstrap with I-BNN mean.
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self._ibnn_value_mean(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value.item()

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            last_values = self._ibnn_value_mean(last_obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=dones,
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ---------------------------------------------------------------------
    # A2C update
    # ---------------------------------------------------------------------

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        buffer = self.rollout_buffer
        assert isinstance(buffer, RolloutBuffer)
        assert buffer.full
        assert not buffer.generator_ready

        # IMPORTANT: estimate uncertainty using the critic that existed while
        # this rollout was collected. Only after the actor update do we condition
        # the critic on this rollout. This avoids artificially tiny uncertainty
        # from evaluating the GP on points it was just trained on.
        advantage_std = self._advantage_std_from_current_critic()

        observations_np = RolloutBuffer.swap_and_flatten(buffer.observations)
        actions_np = RolloutBuffer.swap_and_flatten(buffer.actions).astype(
            np.float32, copy=False
        )
        advantages_np = RolloutBuffer.swap_and_flatten(buffer.advantages).reshape(-1)
        returns_np = RolloutBuffer.swap_and_flatten(buffer.returns).reshape(-1)
        old_values_np = RolloutBuffer.swap_and_flatten(buffer.values).reshape(-1)

        observations = th.as_tensor(observations_np, device=self.device)
        actions = th.as_tensor(actions_np, device=self.device)
        advantages = th.as_tensor(advantages_np, device=self.device, dtype=th.float32)
        returns = th.as_tensor(returns_np, device=self.device, dtype=th.float32)
        old_values = th.as_tensor(old_values_np, device=self.device, dtype=th.float32)

        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        # Width between mu-k*sigma and mu+k*sigma is 2*k*sigma.
        uncertainty_width = 2.0 * self.confidence_std_mult * advantage_std
        scaled_advantages = advantages / (uncertainty_width + self.uncertainty_eps)

        if self.max_scaled_advantage is not None:
            scaled_advantages = th.clamp(
                scaled_advantages,
                -self.max_scaled_advantage,
                self.max_scaled_advantage,
            )

        if self.normalize_advantage:
            scaled_advantages = (
                scaled_advantages - scaled_advantages.mean()
            ) / (scaled_advantages.std() + 1e-8)

        # Actor-only evaluation. Do not call evaluate_actions(), because that
        # would also run SB3's unused neural value branch.
        distribution = self.policy.get_distribution(observations)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        policy_loss = -(scaled_advantages.detach() * log_prob).mean()

        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        # The GP critic is trained separately below; there is no neural value loss
        # in the policy optimizer.
        loss = policy_loss + self.ent_coef * entropy_loss

        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        # Diagnostics for the pre-update critic used to produce the rollout.
        value_loss = th.nn.functional.mse_loss(old_values, returns)
        explained_var = explained_variance(old_values_np, returns_np)

        # Now train/condition the I-BNN critic on this rollout's lambda-return
        # targets, so the updated critic is used on the NEXT rollout.
        critic_x = self._obs_tensor_to_gp_x(observations)
        self._fit_ibnn_critic(critic_x, returns)

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_mse_pre_update", value_loss.item())
        self.logger.record("train/advantage_std_mean", advantage_std.mean().item())
        self.logger.record("train/uncertainty_width_mean", uncertainty_width.mean().item())
        self.logger.record(
            "train/scaled_advantage_abs_mean",
            scaled_advantages.abs().mean().item(),
        )
        self.logger.record("train/ibnn_train_points", self._gp_train_x.shape[0])
        self.logger.record("train/ibnn_weight_var", self.ibnn_weight_var)
        self.logger.record("train/ibnn_bias_var", self.ibnn_bias_var)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
