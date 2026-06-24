import json
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OODMetrics:
    """Tracks and reports out-of-distribution evaluation metrics across episodes."""

    total_terminals: int = 0
    total_timeout: int = 0
    total_base_contact: int = 0
    total_bad_orientation: int = 0
    episode_vel_rewards: list = field(default_factory=list)

    def update(self, dones, info, termination_mgr) -> None:
        """Call once per sim step to accumulate terminal statistics."""
        base_contact = termination_mgr.get_term("base_contact")
        bad_orientation = termination_mgr.get_term("bad_orientation")
        time_out = termination_mgr.get_term("time_out")

        self.total_terminals += int(dones.sum().item())
        self.total_base_contact += int((dones & base_contact).sum().item())
        self.total_bad_orientation += int((dones & bad_orientation).sum().item())
        self.total_timeout += int((dones & time_out).sum().item())

        if dones.any():
            vel_reward = info["log"]["Episode_Reward/track_lin_vel_xy"]
            self.episode_vel_rewards.append(vel_reward.mean().item())

    @property
    def computed_fractions(self) -> dict | None:
        """Returns per-cause terminal fractions, or None if no episodes completed."""
        if self.total_terminals == 0:
            return None
        n = self.total_terminals
        return {
            "timeout_fraction": self.total_timeout / n,
            "base_contact_fraction": self.total_base_contact / n,
            "bad_orientation_fraction": self.total_bad_orientation / n,
            "avg_vel_reward": (
                sum(self.episode_vel_rewards) / len(self.episode_vel_rewards)
                if self.episode_vel_rewards else None
            ),
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout. No-ops if no episodes completed."""
        fracs = self.computed_fractions
        if fracs is None:
            print("[OOD] No episodes completed — nothing to report.")
            return

        print("\n===== OOD Metrics =====")
        print(f"Episodes completed:      {self.total_terminals}")
        print(f"Velocity tracking reward:{fracs['avg_vel_reward']:.4f}")
        print(f"Timeout fraction:        {fracs['timeout_fraction']:.4f}")
        print(f"Base-contact fraction:   {fracs['base_contact_fraction']:.4f}")
        print(f"Bad-orientation fraction:{fracs['bad_orientation_fraction']:.4f}")

    def save(self, task_name: str, description: str, output_dir: str = "Metrics") -> str:
        """
        Persist metrics to a timestamped JSON file.

        Returns the path the file was written to.
        """
        fracs = self.computed_fractions or {}
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results = {
            "description": description,
            "timestamp": timestamp,
            "total_terminals": self.total_terminals,
            "total_timeout": self.total_timeout,
            "total_base_contact": self.total_base_contact,
            "total_bad_orientation": self.total_bad_orientation,
            **fracs,
        }

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"ood_metrics_{task_name}_{timestamp}.json")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        return save_path