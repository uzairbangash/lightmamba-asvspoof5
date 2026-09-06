# """
# Utilization functions
# """

# import os
# import random
# import sys

# import numpy as np
# import torch


# def str_to_bool(val):
#     """Convert a string representation of truth to true (1) or false (0).
#     Copied from the python implementation distutils.utils.strtobool

#     True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
#     are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
#     'val' is anything else.
#     >>> str_to_bool('YES')
#     1
#     >>> str_to_bool('FALSE')
#     0
#     """
#     val = val.lower()
#     if val in ('y', 'yes', 't', 'true', 'on', '1'):
#         return True
#     if val in ('n', 'no', 'f', 'false', 'off', '0'):
#         return False
#     raise ValueError('invalid truth value {}'.format(val))


# def cosine_annealing(step, total_steps, lr_max, lr_min):
#     """Cosine Annealing for learning rate decay scheduler"""
#     return lr_min + (lr_max -
#                      lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# def keras_decay(step, decay=0.0001):
#     """Learning rate decay in Keras-style"""
#     return 1. / (1. + decay * step)


# class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
#     """SGD with restarts scheduler"""
#     def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
#         self.Ti = T0
#         self.T_mul = T_mul
#         self.eta_min = eta_min

#         self.last_restart = 0

#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         T_cur = self.last_epoch - self.last_restart
#         if T_cur >= self.Ti:
#             self.last_restart = self.last_epoch
#             self.Ti = self.Ti * self.T_mul
#             T_cur = 0

#         return [
#             self.eta_min + (base_lr - self.eta_min) *
#             (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
#             for base_lr in self.base_lrs
#         ]


# def _get_optimizer(model_parameters, optim_config):
#     """Defines optimizer according to the given config"""
#     optimizer_name = optim_config['optimizer']

#     if optimizer_name == 'sgd':
#         optimizer = torch.optim.SGD(model_parameters,
#                                     lr=optim_config['base_lr'],
#                                     momentum=optim_config['momentum'],
#                                     weight_decay=optim_config['weight_decay'],
#                                     nesterov=optim_config['nesterov'])
#     elif optimizer_name == 'adam':
#         optimizer = torch.optim.Adam(model_parameters,
#                                      lr=optim_config['base_lr'],
#                                      betas=optim_config['betas'],
#                                      weight_decay=optim_config['weight_decay'],
#                                      amsgrad=str_to_bool(
#                                          optim_config['amsgrad']))
#     else:
#         print('Un-known optimizer', optimizer_name)
#         sys.exit()

#     return optimizer


# def _get_scheduler(optimizer, optim_config):
#     """
#     Defines learning rate scheduler according to the given config
#     """
#     if optim_config['scheduler'] == 'multistep':
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             milestones=optim_config['milestones'],
#             gamma=optim_config['lr_decay'])

#     elif optim_config['scheduler'] == 'sgdr':
#         scheduler = SGDRScheduler(optimizer, optim_config['T0'],
#                                   optim_config['Tmult'],
#                                   optim_config['lr_min'])

#     elif optim_config['scheduler'] == 'cosine':
#         total_steps = optim_config['epochs'] * \
#             optim_config['steps_per_epoch']

#         scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer,
#             lr_lambda=lambda step: cosine_annealing(
#                 step,
#                 total_steps,
#                 1,  # since lr_lambda computes multiplicative factor
#                 optim_config['lr_min'] / optim_config['base_lr']))

#     elif optim_config['scheduler'] == 'keras_decay':
#         scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer, lr_lambda=lambda step: keras_decay(step))
#     else:
#         scheduler = None
#     return scheduler


# def create_optimizer(model_parameters, optim_config):
#     """Defines an optimizer and a scheduler"""
#     optimizer = _get_optimizer(model_parameters, optim_config)
#     scheduler = _get_scheduler(optimizer, optim_config)
#     return optimizer, scheduler


# def seed_worker(worker_id):
#     """
#     Used in generating seed for the worker of torch.utils.data.Dataloader
#     """
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# def set_seed(seed, config = None):
#     """ 
#     set initial seed for reproduction
#     """
#     if config is None:
#         raise ValueError("config should not be None")

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
#         torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])




# below is updated version for wavlm and light transoformer
"""
Utility helpers for the v2 ASVspoof training pipeline.

This module keeps the familiar public helpers from ``utils.py`` while adding:
    - stronger type hints
    - safer validation and error messages
    - AdamW support
    - cleaner scheduler construction
    - reproducibility helpers that work well with newer PyTorch versions
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch


# ============================================================
# Constants
# ============================================================
TRUE_STRINGS = {"y", "yes", "t", "true", "on", "1"}
FALSE_STRINGS = {"n", "no", "f", "false", "off", "0"}
BoolLike = Union[str, bool, int]


# ============================================================
# Boolean / config helpers
# ============================================================
def str_to_bool(value: BoolLike) -> bool:
    """Convert common truthy / falsy values to ``bool``.

    This is backward-compatible with the original helper, but now also accepts
    real booleans and integers directly.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"invalid integer truth value {value}")

    normalized = str(value).strip().lower()
    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False
    raise ValueError(f"invalid truth value {value}")


def get_config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    """Read a config value with a backward-compatible default."""

    return config[key] if key in config else default


# ============================================================
# Learning-rate schedule helpers
# ============================================================
def cosine_annealing(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """Cosine annealing learning-rate value."""

    if total_steps <= 0:
        return lr_min
    clipped_step = min(max(step, 0), total_steps)
    cosine_term = 0.5 * (1.0 + np.cos(clipped_step / total_steps * np.pi))
    return float(lr_min + (lr_max - lr_min) * cosine_term)


def keras_decay(step: int, decay: float = 0.0001) -> float:
    """Keras-style inverse-time decay."""

    return float(1.0 / (1.0 + decay * step))


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine scheduler with warm restarts."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t0: int,
        t_mul: int,
        eta_min: float,
        last_epoch: int = -1,
    ) -> None:
        if t0 <= 0:
            raise ValueError("t0 must be > 0 for SGDRScheduler")
        if t_mul <= 0:
            raise ValueError("t_mul must be > 0 for SGDRScheduler")

        self.ti = t0
        self.t_mul = t_mul
        self.eta_min = eta_min
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        t_cur = self.last_epoch - self.last_restart
        if t_cur >= self.ti:
            self.last_restart = self.last_epoch
            self.ti = self.ti * self.t_mul
            t_cur = 0

        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1.0 + np.cos(np.pi * t_cur / self.ti)) / 2.0
            for base_lr in self.base_lrs
        ]


# ============================================================
# Optimizer / scheduler builders
# ============================================================
def _normalize_betas(betas: Sequence[float]) -> tuple[float, float]:
    if len(betas) != 2:
        raise ValueError("optimizer betas must contain exactly two values")
    return float(betas[0]), float(betas[1])


def _get_optimizer(
    model_parameters: Any,
    optim_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Build an optimizer from config."""

    optimizer_name = str(optim_config["optimizer"]).strip().lower()
    base_lr = float(optim_config["base_lr"])
    weight_decay = float(optim_config.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model_parameters,
            lr=base_lr,
            momentum=float(optim_config.get("momentum", 0.0)),
            weight_decay=weight_decay,
            nesterov=str_to_bool(optim_config.get("nesterov", False)),
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model_parameters,
            lr=base_lr,
            betas=_normalize_betas(optim_config.get("betas", (0.9, 0.999))),
            weight_decay=weight_decay,
            amsgrad=str_to_bool(optim_config.get("amsgrad", False)),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model_parameters,
            lr=base_lr,
            betas=_normalize_betas(optim_config.get("betas", (0.9, 0.999))),
            weight_decay=weight_decay,
            amsgrad=str_to_bool(optim_config.get("amsgrad", False)),
        )

    raise ValueError(f"unknown optimizer '{optim_config['optimizer']}'")


def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    optim_config: Mapping[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build a scheduler from config."""

    scheduler_name = str(optim_config.get("scheduler", "none")).strip().lower()

    if scheduler_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(optim_config["milestones"]),
            gamma=float(optim_config["lr_decay"]),
        )

    if scheduler_name == "sgdr":
        return SGDRScheduler(
            optimizer,
            t0=int(optim_config["T0"]),
            t_mul=int(optim_config["Tmult"]),
            eta_min=float(optim_config["lr_min"]),
        )

    if scheduler_name == "cosine":
        total_steps = int(optim_config["epochs"]) * int(optim_config["steps_per_epoch"])
        lr_min = float(optim_config["lr_min"])
        base_lr = float(optim_config["base_lr"])

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step=step,
                total_steps=total_steps,
                lr_max=1.0,
                lr_min=lr_min / max(base_lr, 1e-12),
            ),
        )

    if scheduler_name == "keras_decay":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: keras_decay(step),
        )

    if scheduler_name in {"none", "null", ""}:
        return None

    raise ValueError(f"unknown scheduler '{optim_config['scheduler']}'")


def create_optimizer(
    model_parameters: Any,
    optim_config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    """Create an optimizer and its scheduler."""

    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


# ============================================================
# Reproducibility helpers
# ============================================================
def seed_worker(worker_id: int) -> None:
    """Seed a ``DataLoader`` worker process deterministically."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int, config: Mapping[str, Any] | None = None) -> None:
    """Set Python / NumPy / PyTorch random seeds.

    ``config`` is optional for better reuse. If provided, the original CUDNN
    toggles are respected.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if config is not None:
            torch.backends.cudnn.deterministic = str_to_bool(
                get_config_value(config, "cudnn_deterministic_toggle", "True")
            )
            torch.backends.cudnn.benchmark = str_to_bool(
                get_config_value(config, "cudnn_benchmark_toggle", "False")
            )


# ============================================================
# Small inspection helpers
# ============================================================
def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)``."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)
