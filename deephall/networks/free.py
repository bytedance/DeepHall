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


import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from scipy import special as ss

from deephall.config import System


def make_monopole_harm(q, l, m):  # NOQA
    norm_factor = np.sqrt(
        ((2 * l + 1) / (4 * np.pi))
        * (ss.factorial(l - m) * ss.factorial(l + m))
        / (ss.factorial(l - q) * ss.factorial(l + q))
    )
    s = np.arange(l - m + 1)
    sum_factors = jnp.array(
        (-1) ** (l - m - s) * ss.comb(l - q, s) * ss.comb(l + q, l - m - s)
    )

    def Y_qlm(electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        x = jnp.cos(theta)
        theta_part = jnp.sum(
            sum_factors
            * (1 - x[..., None]) ** (l - s - (m + q) / 2)
            * (1 + x[..., None]) ** (s + (m + q) / 2),
            axis=-1,
        )
        return norm_factor / 2**l * theta_part * jnp.exp(1j * m * phi)

    return Y_qlm


class Free(nn.Module):
    system: System

    def setup(self):
        orbitals = []
        remaining_elec = sum(self.system.nspins)
        m = ell = q = self.system.flux / 2
        while remaining_elec > 0:
            orbitals.append(make_monopole_harm(q, ell, m))
            remaining_elec -= 1
            if (m := m - 1) < -ell:
                m = ell = ell + 1
        self.orbitals = orbitals

    def __call__(self, electrons):
        orbitals = jnp.stack([phi(electrons) for phi in self.orbitals])
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
