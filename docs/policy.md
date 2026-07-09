# Policy Architecture & Training Configuration

---

## 1. Actor–Critic Architecture

The locomotion policy uses a separate actor and critic network, both implemented as fully connected networks.

| Property | Value |
|---|---|
| Hidden layers | [256, 128, 64] |
| Activation | ELU |
| Policy output | Gaussian action distribution |
| Initial exploration std (σ) | 1.0 |

**ELU activation:**

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ e^x - 1 & x \leq 0 \end{cases}
$$

The critic additionally receives privileged observations (base linear velocity, joint efforts) not available to the policy at deployment time.

---

## 2. LiDAR Latent Observation

The policy observation contains the raw normalized LiDAR scan, but the actor does not feed that full scan directly into the action MLP. A custom actor-critic module encodes the LiDAR slice into a compact latent inside the actor, then concatenates that latent with the proprioceptive observations before computing actions.

| Property | Value |
|---|---|
| Raw LiDAR source | `mdp.lidar` / `RayCasterCfg` sensor named `lidar` |
| Policy observation term | `mdp.lidar` |
| Actor-critic class | `ActorCriticLidarEncoder` |
| Encoder | MLP `[input_dim, 256, 128, 64]` |
| Latent dimension | 64 |
| Activation | ELU |
| Policy history length | 5 |
| Raw LiDAR rays | 1152 values/frame |
| Raw LiDAR history | 5760 values |

The raw LiDAR term is included as the final policy observation term:

```python
lidar = ObsTerm(
    func=mdp.lidar,
    params={
        "sensor_cfg": SceneEntityCfg("lidar"),
        "normalize": True,
    },
    clip=(0.0, 1.0),
)
```

The actor-side encoder is implemented in `source/trakr_rl/trakr_rl/tasks/locomotion/agents/lidar_actor_critic.py`. During the PPO update, RSL-RL samples raw observations from the rollout buffer and calls the actor again. The actor then recomputes:

```text
raw actor observation
-> split proprioception and LiDAR history
-> LiDAR encoder
-> proprioception + LiDAR latent
-> actor MLP
-> action distribution
```

Because the encoder is part of the actor-critic module, its parameters are included in PPO's optimizer automatically and receive gradients through the policy loss:

```text
PPO policy loss -> actor MLP -> LiDAR latent -> LiDAR encoder weights
```

The custom actor is selected in `source/trakr_rl/trakr_rl/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`:

```python
policy = RslRlPpoActorCriticLidarEncoderCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)
```

The critic still receives LiDAR through the critic observation group. The deployed policy receives raw LiDAR in its observation vector, but the exported network uses only the learned latent internally before computing actions.

---

## 3. PPO Hyperparameters

Training uses the [RSL-RL](https://github.com/leggedrobotics/rsl_rl) on-policy PPO implementation.

| Hyperparameter | Value |
|---|---|
| Rollout horizon | 32 steps/env |
| Learning epochs | 5 |
| Mini-batches per update | 4 |
| Clipping parameter (ε) | 0.2 |
| Discount factor (γ) | 0.99 |
| GAE parameter (λ) | 0.95 |
| Initial learning rate (α) | 1e-3 |
| KL divergence target | 0.01 |
| Value loss coefficient | 0.75 |
| Entropy coefficient | 0.01 |
| Gradient clip threshold | 1.0 |

### PPO Clipped Objective

$$
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t\right)\right]
$$

### Generalized Advantage Estimation

$$
\gamma = 0.99, \quad \lambda = 0.95
$$

---

## 4. Runner Configuration

The PPO runner is defined in `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`.

```python
# rsl_rl_ppo_cfg.py
class BasePPORunnerCfg(OnPolicyRunnerCfg):
    num_steps_per_env = 32
    save_interval = 100          # checkpoint every 100 iterations

    policy = ActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = PPOCfg(
        value_loss_coef=0.75,
        entropy_coef=0.01,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

---

## 5. Checkpoint Output

Trained checkpoints and exported policies are saved to:

```
logs/rsl_rl/[TASK-NAME]/
```

This integrates with the same export workflow used by the original `unitree_rl_lab` environments.
