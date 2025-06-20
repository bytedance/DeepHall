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
from functools import partial

import jax
import jax.numpy as jnp
from chex import ArrayTree
from jax.numpy import cos, sin, tan

from deephall.config import InteractionType, System
from deephall.types import AngularMomenta, LocalEnergy, LogPsiNetwork, OtherObservables


def coulomb_potential(cos12: jnp.ndarray, Q: float, r: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-electron Coulomb potential.

    Args:
        cos12: The cosine of the angle between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength. Unused.
        r: Sphere radius.

    Returns:
        potential energy
    """
    del Q
    r_ee = jnp.sqrt(2 - 2 * cos12)
    return jnp.sum(jnp.triu(1 / r_ee, k=1)) / r


def harmonic_potential(cos12: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Returns the simple harmonic potential.

    The word "harmonic" describes the form of the Haldane pseudopotential on LLL:
        V(L) = L(L+1) / 2Q(Q+1) / sqrt(Q)
    and the corresponding real space form is:
        V(theta_12) = 1 + (Q+1) / Q * cos theta_12

    Args:
        cos12: The cosine of the angle between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength.

    Returns:
        potential energy
    """
    return jnp.sum(jnp.triu(1 + (Q + 1) / Q * cos12, k=1))


def make_potential(
    interaction_type: InteractionType, Q: float, r: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q, r=r)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: jnp.ndarray) -> jnp.ndarray:
        theta, phi = data[..., 0], data[..., 1]
        xyz_data = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        cos12 = jnp.einsum("ia,ja->ij", xyz_data, xyz_data)
        return potential_function(cos12)

    return potential


def make_local_kinetic_energy(f: LogPsiNetwork, Q: float, r: jnp.ndarray):
    r"""Creates a function to for the local kinetic energy.

    Args:
        f: Callable which evaluates the log of the magnitude of the wavefunction.
        Q: Monopole strength
        r: Sphere radius

    Returns:
        Callable that evaluates the local kinetic energy, \frac{|\Lambda|^2 f}{2 R^2 f},
        where
            \frac{|\Lambda|^2 f}{f} = -R^2 \frac{\nabla^2 f}{f} + (Q \cot \theta)^2
                + 2i Q \frac{\cot \theta}{\sin \theta} \frac{\partial f}{\partial \phi},
        and
            -\frac{\nabla^2 f}{f} = - [\nabla^2 \log f + (\nabla \log f)^2].
    """

    def _lapl_over_f(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, AngularMomenta]:
        theta, phi = data[..., 0], data[..., 1]

        #        +----------------------------------------------------------+
        #        |           Prepare first and second detivatives           |
        #        +----------------------------------------------------------+

        grad_real = jax.grad(lambda p, x: f(p, x).real, argnums=1)(params, data)
        grad_imag = jax.grad(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        grad_theta = grad_real[..., 0] + 1j * grad_imag[..., 0]
        grad_phi = grad_real[..., 1] + 1j * grad_imag[..., 1]
        # $(\nabla \log \psi) \cdot (\nabla \log \psi)$ on a sphere
        square_grad_logpsi = jnp.sum(grad_theta**2 + grad_phi**2 / sin(theta) ** 2)

        hess_real = jax.hessian(lambda p, x: f(p, x).real, argnums=1)(params, data)
        hess_imag = jax.hessian(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        hess_logpsi = hess_real + 1j * hess_imag

        #        +----------------------------------------------------------+
        #        |                Calculating kinetic energy                |
        #        +----------------------------------------------------------+

        # $\nabla^2 \log \psi$ on a sphere
        grad_grad_logpsi = jnp.sum(
            grad_theta / tan(theta)
            + jnp.diagonal(hess_logpsi[:, 0, :, 0])
            + jnp.diagonal(hess_logpsi[:, 1, :, 1]) / sin(theta) ** 2
        )
        # See section 3.10.3 of "Composite Fermions"
        magnetic_contribution = jnp.sum(
            (Q / tan(theta)) ** 2 + 2j * Q * cos(theta) / sin(theta) ** 2 * grad_phi
        )
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2 / r**2

        #        +----------------------------------------------------------+
        #        |        Calculating angular momentum square (L^2)         |
        #        +----------------------------------------------------------+

        i = (Ellipsis, slice(None), jnp.newaxis)  # same as [..., :, None]
        j = (Ellipsis, jnp.newaxis, slice(None))  # same as [..., None, :]
        r_hat = jnp.stack([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
        phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)])
        theta_hat_prime = jnp.stack(  # Rescaled theta_hat with 1/sin(theta)
            [cos(phi) / tan(theta), sin(phi) / tan(theta), -jnp.ones_like(theta)]
        )
        hess_theta_theta = hess_logpsi[:, 0, :, 0] + grad_theta[*i] * grad_theta[*j]
        hess_theta_phi = hess_logpsi[:, 0, :, 1] + grad_theta[*i] * grad_phi[*j]
        hess_phi_phi = hess_logpsi[:, 1, :, 1] + grad_phi[*i] * grad_phi[*j]
        # Note that theta_hat_prime alrealdy has a 1/sin factor
        magnetic_term = Q * (theta_hat_prime * cos(theta) + r_hat)
        # We first assume everything commutes, and add back extra terms at the end
        angular_momentum_square = jnp.sum(
            2 * phi_hat[*i] * theta_hat_prime[*j] * hess_theta_phi
            - phi_hat[*i] * phi_hat[*j] * hess_theta_theta
            - (theta_hat_prime[*i] * theta_hat_prime[*j] * hess_phi_phi)
            - (2j * magnetic_term[*j])
            * (phi_hat[*i] * grad_theta[*i] - theta_hat_prime[*i] * grad_phi[*i])
            + magnetic_term[*i] * magnetic_term[*j],
        ) - jnp.sum(grad_theta / tan(theta))  # Diagonal extra terms

        #        +----------------------------------------------------------+
        #        |                     Assemble outputs                     |
        #        +----------------------------------------------------------+

        other_observables = AngularMomenta(
            angular_momentum_z=jnp.sum(grad_phi).imag,  # same as (-1j * d_phi).real
            angular_momentum_z_square=-jnp.sum(hess_phi_phi).real,
            angular_momentum_square=angular_momentum_square.real,
        )
        return kinetic_energy, other_observables

    return _lapl_over_f


def local_energy(f: LogPsiNetwork, system: System) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux / 2
    radius = jnp.array(system.radius or jnp.sqrt(Q))
    ke = make_local_kinetic_energy(f, Q, radius)
    pe = make_potential(system.interaction_type, Q, radius)

    def _e_l(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the total energy.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            Local energy and other observables.
        """
        potential = pe(data) * system.interaction_strength
        kinetic, angular_momenta = ke(params, data)
        return kinetic + potential, angular_momenta | {
            "potential": potential,
            "kinetic": kinetic,
        }

    return _e_l
