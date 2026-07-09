from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class LidarActor(nn.Module):
    """Actor module that encodes the final LiDAR slice before the action MLP."""

    def __init__(
        self,
        input_dim: int,
        lidar_dim: int,
        lidar_latent_dim: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int],
        lidar_encoder_hidden_dims: tuple[int] | list[int],
        activation: str,
        state_dependent_std: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.lidar_dim = lidar_dim
        self.lidar_latent_dim = lidar_latent_dim
        self.lidar_encoder = MLP(lidar_dim, lidar_latent_dim, lidar_encoder_hidden_dims, activation)

        encoded_input_dim = input_dim - lidar_dim + lidar_latent_dim
        if state_dependent_std:
            self.action_mlp = MLP(encoded_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.action_mlp = MLP(encoded_input_dim, num_actions, actor_hidden_dims, activation)

    def __getitem__(self, index: int):
        if index == 0:
            return self
        return self.action_mlp[index]

    @property
    def in_features(self) -> int:
        return self.input_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio_obs = obs[..., :-self.lidar_dim]
        lidar_obs = obs[..., -self.lidar_dim :]
        lidar_latent = self.lidar_encoder(lidar_obs)
        actor_obs = torch.cat([proprio_obs, lidar_latent], dim=-1)
        return self.action_mlp(actor_obs)


class ActorCriticLidarEncoder(nn.Module):
    """RSL-RL actor-critic with actor-side LiDAR encoding."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 128, 64],
        critic_hidden_dims: tuple[int] | list[int] = [256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        lidar_dim: int = 5760,
        lidar_latent_dim: int = 64,
        lidar_encoder_hidden_dims: tuple[int] | list[int] = [256, 128],
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticLidarEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.lidar_dim = lidar_dim
        self.lidar_latent_dim = lidar_latent_dim
        self.state_dependent_std = state_dependent_std

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        if self.lidar_dim <= 0 or self.lidar_dim >= num_actor_obs:
            raise ValueError(
                f"lidar_dim must be in (0, {num_actor_obs}), but got {self.lidar_dim}. "
                "The policy observation is expected to end with the raw LiDAR history."
            )

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.actor = LidarActor(
            input_dim=num_actor_obs,
            lidar_dim=self.lidar_dim,
            lidar_latent_dim=self.lidar_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            lidar_encoder_hidden_dims=lidar_encoder_hidden_dims,
            activation=activation,
            state_dependent_std=self.state_dependent_std,
        )
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor.action_mlp[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor.action_mlp[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor.action_mlp[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def register_custom_rsl_rl_modules() -> None:
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.ActorCriticLidarEncoder = ActorCriticLidarEncoder
