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

from typing import (
    TYPE_CHECKING,
    NamedTuple,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from chex import ArrayTree, PRNGKey
from jax import numpy as jnp
from jaxtyping import Array, PyTree
from optax import OptState

# PyTree from jaxtyping is not friendly to static type checkers
if TYPE_CHECKING:
    Params = ArrayTree
else:
    Params = PyTree[Array, "T"]


class AngularMomenta(TypedDict):
    """Angular momenta is calculated with kinetic energy."""

    angular_momentum_z: jnp.ndarray
    angular_momentum_z_square: jnp.ndarray
    angular_momentum_square: jnp.ndarray


class OtherObservables(AngularMomenta):
    """Every thing else when calculating local energy."""

    kinetic: jnp.ndarray
    potential: jnp.ndarray


class LossStats(OtherObservables):
    energy: jnp.ndarray
    variance: jnp.ndarray


class CheckpointState(NamedTuple):
    params: Params
    data: jnp.ndarray
    opt_state: OptState
    mcmc_width: jnp.ndarray


@runtime_checkable
class LocalEnergy(Protocol):
    def __call__(
        self, params: Params, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the local energy of a Hamiltonian at a configuration.

        Args:
            params: network parameters.
            data: MCMC configuration to evaluate.

        Returns:
            A tuple of
            - Local energy for each walker
            - other observables, such as
                - angular momentum
                - kinetic and potential energy
        """


@runtime_checkable
class LogPsiNetwork(Protocol):
    def __call__(self, params: Params, data: jnp.ndarray) -> jnp.ndarray:
        pass


class TrainingStep(Protocol):
    def __call__(
        self, state: CheckpointState, key: PRNGKey
    ) -> tuple[CheckpointState, LossStats]:
        pass


class TrainingInit(Protocol):
    def __call__(self, params: Params, key: PRNGKey, data: jnp.ndarray):
        pass
