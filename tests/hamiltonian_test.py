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

import jax
import pytest
from jax import numpy as jnp

from deephall import hamiltonian
from deephall.config import System
from deephall.networks.laughlin import Laughlin


def sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def make_lll(nelec: int, Q: int):
    def log_psi(_, data):
        theta, phi = data[..., 0], data[..., 1]
        u = jnp.cos(theta / 2) * jnp.exp(1j * phi / 2)
        v = jnp.sin(theta / 2) * jnp.exp(-1j * phi / 2)
        lll_orb = jnp.stack([u**m * v ** (2 * Q - m) for m in range(nelec)], axis=-1)

        sign, logdet = jnp.linalg.slogdet(lll_orb)
        return logdet + jnp.log(sign)

    return log_psi


laplacian_implementaions = [
    hamiltonian.make_local_kinetic_energy_with_fwdlap,
    hamiltonian.make_local_kinetic_energy_with_hessian,
]


@pytest.mark.parametrize("laplacian_func", laplacian_implementaions)
def test_free_electron(laplacian_func):
    def log_psi(params, data):
        """Sphere harmonics with l=1: $Y_{1m}$."""
        theta, phi = data[..., 0], data[..., 1]
        orb = jnp.stack(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.cos(theta),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )
        sign, logdet = jnp.linalg.slogdet(orb)
        return logdet + jnp.log(sign)

    data = sample(jax.random.PRNGKey(1898), 2, nelec=3)
    laplacian = laplacian_func(log_psi, Q=0, r=1)
    batch_laplacian = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
    ke, other_observables = batch_laplacian(None, data)
    assert jnp.allclose(ke, 3, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_square"], 0, atol=1e-3)


@pytest.mark.parametrize("laplacian_func", laplacian_implementaions)
@pytest.mark.parametrize(
    "nelec,Q,L_square,L_z", [(1, 1, 2, -1), (3, 1, 0, 0), (9, 4, 0, 0)]
)
def test_kinetic_and_angular_momentum(
    laplacian_func, nelec: int, Q: int, L_square: float, L_z: float
):
    data = sample(jax.random.PRNGKey(1898), 2, nelec)
    laplacian = laplacian_func(make_lll(nelec, Q), Q, jnp.sqrt(Q))
    batch_laplacian = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
    ke, other_observables = batch_laplacian(None, data)
    assert jnp.allclose(ke, nelec / 2, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_z"], L_z, atol=1e-3)
    assert jnp.allclose(
        other_observables["angular_momentum_z_square"], L_z**2, atol=1e-3
    )
    assert jnp.allclose(
        other_observables["angular_momentum_square"], L_square, atol=1e-3
    )


@pytest.mark.parametrize("laplacian_func", laplacian_implementaions)
@pytest.mark.parametrize("nelec,Q", [(3, 3), (4, 4.5)])
def test_laughlin_energy(laplacian_func, nelec: int, Q: int):
    data = sample(jax.random.PRNGKey(1898), 2, nelec)
    system = System(nspins=(nelec, 0), flux=2 * Q)
    laughlin = Laughlin(system)
    laplacian = laplacian_func(laughlin.apply, Q, jnp.sqrt(Q))
    params = laughlin.init(jax.random.PRNGKey(1898), data[0])
    batch_laplacian = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
    ke, other_observables = batch_laplacian(params, data)
    assert jnp.allclose(ke, nelec / 2, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_z"], 0, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_z_square"], 0, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_square"], 0, atol=1e-3)
