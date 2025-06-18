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

"""This code is an reimplementation of the Psiformer network (Glehn et al., ICLR 2023).

The input feature is chosen as the Cartesian coordinates of the elelctrons:
    h_one_0 = [cos(theta), sin(theta) cos(phi), sin(theta) sin(phi)]
The feature is then passed through standard Psiformer layers, outputing features h_one.
Afterwards, the features are used to construct the orbitals with the monopole harmonics.
The details for the orbital construction are located in `blocks.py`.
"""

from flax import linen as nn
from jax import numpy as jnp

from deephall.config import System

from .blocks import Jastrow, MonopoleProductOrbitals, PsiformerLayers

__all__ = ["MHPO"]


class MHPO(nn.Module):
    """Monopole harmonics product orbital ansatz using Psiformer as the backbone."""

    system: System
    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    flux_per_elec: int = 0

    def __call__(self, electrons):
        orbitals = self.orbitals(electrons)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    @nn.compact
    def orbitals(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        spins = jnp.array([1] * self.system.nspins[0] + [-1] * self.system.nspins[1])
        h_one = PsiformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim,
        )(electrons, spins)
        reduced_flux = self.system.flux - self.flux_per_elec * (
            sum(self.system.nspins) - 1
        )
        orbitals = MonopoleProductOrbitals(
            Q=reduced_flux / 2,
            nspins=self.system.nspins,
            ndets=self.ndets,
            name="Orbitals",  # for backward compatibility
        )(h_one, theta, phi)
        jastrow = Jastrow(self.system.nspins)(electrons)

        if self.flux_per_elec > 0:
            u = jnp.cos(theta / 2) * jnp.exp(0.5j * phi)
            v = jnp.sin(theta / 2) * jnp.exp(-0.5j * phi)
            # Adding eye to avoid NaN/inf
            element = u * v[..., None] - u[..., None] * v + jnp.eye(u.shape[0])
            jastrow += jnp.sum(jnp.triu(jnp.log(element), k=1)) * self.flux_per_elec

        return jnp.exp(jastrow / sum(self.system.nspins)) * orbitals
