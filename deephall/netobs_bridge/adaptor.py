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

from collections.abc import Callable
from typing import Any, TypedDict

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from netobs.adaptors import NetworkAdaptor, WalkingStep
from omegaconf import OmegaConf
from upath import UPath

from deephall import constants
from deephall.config import Config
from deephall.hamiltonian import (
    make_local_kinetic_energy_with_hessian,
    make_potential,
)
from deephall.log import LogManager
from deephall.mcmc import make_mcmc_step
from deephall.netobs_bridge.hall_system import HallSystem
from deephall.networks import make_network


class DeepHallAuxData(TypedDict):
    mcmc_width: jnp.ndarray


@register_pytree_node_class
class DeepHallAdaptor(NetworkAdaptor[HallSystem]):
    def __init__(self, config: Any, args: list[str]) -> None:
        super().__init__(config, args)

    def restore(
        self, ckpt_file: str | None = None
    ) -> tuple[Any, jnp.ndarray, HallSystem, Any]:
        if ckpt_file is None:
            raise ValueError("Must specify a checkpoint")
        ckpt_path = UPath(ckpt_file)
        config_path = ckpt_path.parent / "config.yml"
        self.cfg = cfg = Config.from_dict(OmegaConf.load(config_path.open()))  # type: ignore
        model = make_network(cfg.system, cfg.network)
        self.network = jax.jit(model.apply)
        self.batch_per_device = cfg.batch_size // jax.device_count()
        Q = cfg.system.flux / 2
        radius = jnp.array(cfg.system.radius or jnp.sqrt(Q))
        self.kinetic_energy = make_local_kinetic_energy_with_hessian(
            self.network, Q, radius
        )
        self.potential_energy = make_potential(cfg.system.interaction_type, Q, radius)
        _, state = LogManager.restore_checkpoint(ckpt_path)

        return (
            state.params,
            state.data,
            HallSystem(spins=list(cfg.system.nspins), ndim=2, flux=cfg.system.flux),
            DeepHallAuxData(mcmc_width=state.mcmc_width),
        )

    def call_signed_network(
        self, params: jnp.ndarray, electrons: jnp.ndarray, system: HallSystem
    ):
        del system
        return jnp.array(1.0), self.network(params, electrons)

    def make_walking_step(
        self, batch_log_psi: Callable, steps: int, system: HallSystem
    ) -> WalkingStep[DeepHallAuxData]:
        del system
        mcmc_step = make_mcmc_step(
            lambda params, data: batch_log_psi(params, data, None),
            batch_per_device=self.batch_per_device,
            steps=steps,
        )

        def walk(
            key: jnp.ndarray,
            params: jnp.ndarray,
            electrons: jnp.ndarray,
            aux_data: DeepHallAuxData,
        ) -> tuple[jnp.ndarray, DeepHallAuxData]:
            new_data, _ = mcmc_step(params, electrons, key, aux_data["mcmc_width"])
            return new_data, aux_data

        return constants.pmap(walk)

    def call_local_kinetic_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: HallSystem,
    ) -> jnp.ndarray:
        del key, system
        return self.kinetic_energy(params, electrons)[0]

    def call_local_potential_energy(
        self,
        params: jnp.ndarray,
        key: jnp.ndarray,
        electrons: jnp.ndarray,
        system: HallSystem,
    ) -> jnp.ndarray:
        del params, system, key
        return self.potential_energy(electrons) * self.cfg.system.interaction_strength


DEFAULT = DeepHallAdaptor
