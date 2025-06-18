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


class FeaturedOrbitals(nn.Module):
    nspins: tuple[int, int]
    features: list[int]

    @nn.compact
    def __call__(self, h_one):
        orbital_list = [
            nn.DenseGeneral(self.features)(h_one_alpha)
            + 1j * nn.DenseGeneral(self.features)(h_one_alpha)
            for h_one_alpha in jnp.split(h_one, (self.nspins[0],))
            if len(h_one_alpha)
        ]
        return jnp.concat(orbital_list)


class Orbitals(nn.Module):
    Q: float
    nspins: tuple[int, int]
    ndets: int

    def setup(self):
        m = np.arange(-self.Q, self.Q + 1)
        self.norm_factor = jnp.array(np.sqrt(ss.comb(2 * self.Q, self.Q - m)))
        self.featured_orbitals = FeaturedOrbitals(
            nspins=self.nspins,
            features=(int(self.Q * 2) + 1, sum(self.nspins), self.ndets),
        )

    def __call__(self, h_one, theta, phi):
        orbitals = self.featured_orbitals(h_one)

        m = jnp.arange(-self.Q, self.Q + 1)
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]
        envelope = self.norm_factor * u ** (self.Q + m) * v ** (self.Q - m)
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

        return jnp.moveaxis(orbitals, -1, 0)  # Move ndets dim to the front


class Jastrow(nn.Module):
    nspins: tuple[int, int]

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        nspins = self.nspins
        r_ee = self.calculated_r_ee(electrons)
        r_ees = [
            jnp.split(r, nspins[0:1], axis=1)
            for r in jnp.split(r_ee, nspins[0:1], axis=0)
        ]
        r_ees_parallel = jnp.concatenate(
            [
                r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
                r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
            ]
        )

        if r_ees_parallel.shape[0] > 0:
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,))
            jastrow_ee_par = jnp.sum(
                -(0.25 * alpha_par**2) / (alpha_par + r_ees_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,))
            jastrow_ee_anti = jnp.sum(
                -(0.5 * alpha_anti**2) / (alpha_anti + r_ees[0][1])
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_anti + jastrow_ee_par

    def calculated_r_ee(self, electrons: jnp.ndarray) -> jnp.ndarray:
        theta, phi = electrons[..., 0], electrons[..., 1]
        cart_e = jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )
        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
