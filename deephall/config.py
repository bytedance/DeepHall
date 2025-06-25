# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Any, Self, TypeVar

T = TypeVar("T")


def from_dict(cls: type[T], dikt: dict[str, Any]) -> T:
    """Restore dataclass from a OmegaConf dictionary.

    Args:
        cls: The class of the dataclass
        dikt: The dictionary containing the properties of the dataclass

    Raises:
        ValueError: the dictionary and the dataclass is not compatible

    Returns:
        The dataclass instance.
    """
    try:
        fieldtypes = {f.name: f.type for f in fields(cls)}  # type: ignore
        return cls(
            **{
                f: from_dict(fieldtypes[f], dikt[f])  # type: ignore
                if is_dataclass(fieldtypes[f])
                else dikt[f]
                for f in dikt
                if f in fieldtypes  # allow extra keys
            }
        )
    except Exception as e:
        raise ValueError(f"Error converting dictionary to {cls.__name__}: {e}")


class InteractionType(StrEnum):
    coulomb = "coulomb"
    harmonic = "harmonic"


@dataclass
class System:
    flux: int = 2
    "Positive or negative integer $2Q$."

    radius: float | None = None
    r"By default, the radius of the sphere is fixed at $\sqrt{Q}$."

    nspins: tuple[int, int] = (3, 0)
    "Number of spin-up and spin-down electrons."

    interaction_strength: float = 1.0
    "The factor for the potential energy."

    lz_center: float = 0.0
    "Lz to pick using penalty method."

    lz_penalty: float = 0.0
    "The strength of the penalty for (Lz - lz_center)^2."

    l2_penalty: float = 0.0
    "The strength of the penalty for L^2."

    interaction_type: InteractionType = InteractionType.coulomb


class NetworkType(StrEnum):
    psiformer = "psiformer"
    laughlin = "laughlin"
    free = "free"


@dataclass
class PsiformerNetwork:
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    determinants: int = 1


@dataclass
class Network:
    type: NetworkType = NetworkType.psiformer
    psiformer: PsiformerNetwork = field(default_factory=PsiformerNetwork)


@dataclass
class MCMC:
    steps: int = 10
    "MCMC steps to run between steps."

    width: float = 0.1
    "The std dev for gaussian move."

    burn_in: int = 200
    """MCMC burn in steps to run before training.

    It's actually `mcmc.burn_in * mcmc.steps` number of steps.
    """

    adapt_frequency: int = 100
    "Number of steps after which to update the adaptive MCMC step size."


@dataclass
class LearningRate:
    """Define the learning rate.

    The formula is rate * (1.0 / (1.0 + (t / delay)) ** decay
    """

    rate: float = 0.005
    decay: float = 1.0
    delay: float = 2000.0

    def schedule(self, t):
        return self.rate * (1.0 / (1.0 + (t / self.delay))) ** self.decay


class OptimizerName(StrEnum):
    adam = "adam"
    kfac = "kfac"
    none = "none"


@dataclass
class OptimizerAdam:
    lr: LearningRate = field(default_factory=LearningRate)


@dataclass
class OptimizerKfac:
    lr: LearningRate = field(default_factory=lambda: LearningRate(rate=0.05))

    norm_constraint: float = 1e-3
    """
    The update is scaled down so that its approximate squared Fisher norm ``v^T F v``
    is at most the specified value.
    """

    curvature_ema: float = 0.95
    "The decay factor used when calculating the covariance estimate moving averages."

    l2_reg: float = 0.0
    "Tell the optimizer what L2 regularization coefficient you are using (if any)."

    inverse_update_period: int = 1
    "The number of steps in between updating the inverse curvature approximation."

    damping: float = 1e-3
    "Fixed damping parameter."


@dataclass
class Optim:
    reset: bool = False
    iterations: int = 1000
    optimizer: OptimizerName | None = OptimizerName.kfac
    adam: OptimizerAdam = field(default_factory=OptimizerAdam)
    kfac: OptimizerKfac = field(default_factory=OptimizerKfac)


@dataclass
class Log:
    save_path: str | None = None
    """Path to save checkpoints and logs.

    Can be any path supported by fsspec/universal_pathlib.
    """

    restore_path: str | None = None
    """
    Path to restore checkpoints.

    Can be a directory containing checkpoints or path to a specific checkpoint.
    """

    save_time_interval: int = 10 * 60
    """Minimum time (in seconds) between checkpoint saves.

    A checkpoint will only be saved if both this interval has passed and
    the current step is a multiple of `save_step_interval`.
    """

    save_step_interval: int = 1000
    """Checkpoints are saved only at steps that are multiples of this value.

    Checkpoints are saved only at steps that are multiples of this value,
    and only if the `save_time_interval` has also elapsed.
    """

    initial_energy: bool = True
    """Log initial energy before any optimizations.

    This is helpful for debugging. If we have initial energy but have error in training,
    it's probably optimizer's fault
    """


class LaplacianMode(StrEnum):
    hessian = "hessian"
    forward = "forward"


@dataclass
class Config:
    batch_size: int = 3360  # 32*3*5*7
    laplacian: LaplacianMode = LaplacianMode.hessian
    seed: int = field(default_factory=lambda: int(time.time()))
    system: System = field(default_factory=System)
    network: Network = field(default_factory=Network)
    mcmc: MCMC = field(default_factory=MCMC)
    optim: Optim = field(default_factory=Optim)
    log: Log = field(default_factory=Log)

    @classmethod
    def from_dict(cls, dikt: dict) -> Self:
        """Convert a dictionary to Config."""
        return from_dict(cls, dikt)
