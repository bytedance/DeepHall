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

from flax import linen as nn
from jax import numpy as jnp

from deephall.config import System


class Laughlin(nn.Module):
    """Create Laughlin wavefunction for ground or quasiparticle/quasihole state."""

    system: System

    cf_flux: int = 1
    "Flux p for composite fermion."

    def setup(self):
        nelec = sum(self.system.nspins)
        self.Q1 = self.system.flux / 2 - self.cf_flux * (sum(self.system.nspins) - 1)
        if nelec == 2 * self.Q1 + 1:  # Ground state
            self.cf_orbitals = self.full_orbitals
        elif nelec == 2 * self.Q1:  # Quasihole
            self._check_lz()
            assert -abs(self.Q1) <= self.system.excitation_lz <= abs(self.Q1)
            self.cf_orbitals = self.quasihole_orbitals
        elif nelec == 2 * self.Q1 + 2:  # Quasiparitcle
            self._check_lz()
            assert -abs(self.Q1) - 1 <= self.system.excitation_lz <= abs(self.Q1) + 1
            self.cf_orbitals = self.quasihole_orbitals
            self.cf_orbitals = self.quasiparticle_orbitals
        else:
            raise ValueError("Filling not supported")

    def _check_lz(self):
        """Make sure the specified Lz is possible for quasiparticle/quasihole state."""
        diff = self.system.excitation_lz - self.Q1
        assert (
            int(diff) == diff
        ), f"Impossible Lz={self.system.excitation_lz} for excitation"

    def __call__(self, electrons):
        orbitals = self.orbitals(electrons)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    def orbitals(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]

        return self.cf_orbitals(u, v)

    def full_orbitals(self, u, v):
        Q = self.Q1
        m = jnp.arange(-Q, Q + 1)
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        return u ** (Q + m) * v ** (Q - m) * jastrow

    def quasihole_orbitals(self, u, v):
        Q = self.Q1
        m = jnp.concat(
            [
                jnp.arange(-Q, -self.system.excitation_lz),
                jnp.arange(Q, -self.system.excitation_lz, -1),
            ]
        )
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        return u ** (Q + m) * v ** (Q - m) * jastrow

    def quasiparticle_orbitals(self, u, v):
        Q = self.Q1
        m = jnp.arange(-Q, Q + 1)
        orbitals = u ** (Q + m) * v ** (Q - m)

        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        # LLL projection (u* -> \partial_u, v* -> \partial_v)
        jastrow_dv = jastrow * (jnp.sum(-u[:, 0] / element, axis=-1, keepdims=True) + u)
        jastrow_du = jastrow * (jnp.sum(v[:, 0] / element, axis=-1, keepdims=True) - v)

        m1 = self.system.excitation_lz
        excited = (u ** (Q + m1) * v ** (Q - m1)) * (
            (Q + 1 + m1) * v * jastrow_dv - (Q + 1 - m1) * u * jastrow_du
        )
        return jnp.concat([orbitals * jastrow, excited], axis=-1)
