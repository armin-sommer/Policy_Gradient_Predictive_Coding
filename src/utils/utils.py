from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    env_name: str = "coinrun"
    num_train_levels: int = 200
    num_test_levels: int = 0
    distribution_mode: str = "easy"
    num_envs: int = 64
    # bandit-only options (ignored by Procgen)
    arm_means: tuple = (1.0, 0.9)
    deterministic_rewards: bool = True


@dataclass
class LoggingConfig:
    log_dir: str = "outputs"
    log_interval: int = 1
    save_interval: int = 100


@dataclass
class WandbConfig:
    mode: str = "disabled"
    project: str = "procgen-pcpg"
    group: Optional[str] = None


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42