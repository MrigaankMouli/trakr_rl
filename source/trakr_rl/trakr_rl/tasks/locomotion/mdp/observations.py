from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def lidar(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        normalize: bool = True
) -> torch.Tensor:
    
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w

    origins = sensor.data.pos_w.unsqueeze(1)

    distances = torch.linalg.norm(
        hits - origins,
        dim = -1
    )

    distances = torch.nan_to_num(
        distances,
        nan=sensor.cfg.max_distance,
    )

    if normalize:
        distances = (
            distances /
            sensor.cfg.max_distance
        )

    return distances