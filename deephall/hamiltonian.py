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

from functools import partial

import folx
import jax
import jax.numpy as jnp
from folx.api import FwdLaplArray
from jax.numpy import cos, sin, tan
from jaxtyping import Array, Complex, Float

from deephall.config import InteractionType, LaplacianMode, System
from deephall.types import (
    AngularMomenta,
    LocalEnergy,
    LogPsiNetwork,
    OtherObservables,
    Params,
)


def coulomb_potential(
    cos12: Float[Array, "nelec nelec"],
    Q: float | int,
    r: float | int | Float[Array, ""],
) -> Float[Array, ""]:
    """Returns the electron-electron Coulomb potential.

    Args:
        cos12: The cosine of the angle between two electrons.
        Q: Monopole strength. Unused.
        r: Sphere radius.

    Returns:
        potential energy
    """
    del Q
    r_ee = jnp.sqrt(2 - 2 * cos12)
    return jnp.sum(jnp.triu(1 / r_ee, k=1)) / r


def harmonic_potential(
    cos12: Float[Array, "nelec nelec"], Q: float | int
) -> Float[Array, ""]:
    """Returns the simple harmonic potential.

    The word "harmonic" describes the form of the Haldane pseudopotential on LLL:
        V(L) = L(L+1) / 2Q(Q+1) / sqrt(Q)
    and the corresponding real space form is:
        V(theta_12) = 1 + (Q+1) / Q * cos theta_12

    Args:
        cos12: The cosine of the angle between two electrons.
        Q: Monopole strength.

    Returns:
        potential energy
    """
    return jnp.sum(jnp.triu(1 + (Q + 1) / Q * cos12, k=1))


def make_potential(
    interaction_type: InteractionType, Q: float | int, r: float | int | Float[Array, ""]
):
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q, r=r)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: Float[Array, "nelec 2"]) -> Float[Array, ""]:
        theta, phi = data[..., 0], data[..., 1]
        xyz_data: Float[Array, "nelec 3"] = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        cos12: Float[Array, "nelec nelec"] = jnp.einsum("ia,ja->ij", xyz_data, xyz_data)
        return potential_function(cos12)

    return potential


def make_local_kinetic_energy_with_hessian(
    f: LogPsiNetwork, Q: float | int, r: float | int | Float[Array, ""]
):
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
        params: Params, data: Float[Array, "nelec 2"]
    ) -> tuple[Complex[Array, ""], AngularMomenta]:
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


def make_local_kinetic_energy_with_fwdlap(
    f: LogPsiNetwork, Q: float | int, r: float | int | Float[Array, ""]
):
    angular_momentum = make_local_angular_momentum_square(f, Q)

    def _lapl_over_f(
        params: Params, data: Float[Array, "nelec 2"]
    ) -> tuple[Complex[Array, ""], AngularMomenta]:
        theta = data[..., 0]

        fwd_f = folx.forward_laplacian(lambda x: f(params, x))
        fwdlap_weights = jnp.stack([jnp.ones_like(theta), 1 / sin(theta)], axis=-1)
        fwdlap_output: FwdLaplArray = fwd_f(data, weights=fwdlap_weights)
        grad_logpsi = fwdlap_output.dense_jacobian.reshape(data.shape) / fwdlap_weights
        grad_theta, grad_phi = grad_logpsi[..., 0], grad_logpsi[..., 1]
        # $(\nabla \log \psi) \cdot (\nabla \log \psi)$ on a sphere
        square_grad_logpsi = jnp.sum(grad_theta**2 + grad_phi**2 / sin(theta) ** 2)

        # $\nabla^2 \log \psi$ on a sphere
        grad_grad_logpsi = jnp.sum(grad_theta / tan(theta)) + fwdlap_output.laplacian
        # See section 3.10.3 of "Composite Fermions"
        magnetic_contribution = jnp.sum(
            (Q / tan(theta)) ** 2 + 2j * Q * cos(theta) / sin(theta) ** 2 * grad_phi
        )
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2 / r**2

        other_observables = angular_momentum(params, data)
        return kinetic_energy, other_observables

    return _lapl_over_f


def make_angular_momentum_operator(f: LogPsiNetwork, Q: float | int):
    r"""Create angular momentum operator $\hat L$.

    Following section 3.10 of the book "Composite Fermions", $\hat L$ is defined as:

    \hat L = - i \hat\phi \partial_\theta + \hat\theta \frac{i}{\sin\theta}\partial_\phi
        + Q (\cot\theta \hat\theta + \hat r)

    where \hat\phi, \hat\theta, \hat r are unit vectors on the sphere, and please refer
    to the "Composite Fermions" book for their definitions.

    Different from the case without magnetic fields, the angular momentum operator here
    has an additional "constant" term that does not involve derivatives. While this may
    not be a big issue for the angular momentum operator itself, it does poses some
    challenges when evaluating the angular momentum square operator. Therefore, the
    function we make here returns the results for the differential term and the magnetic
    term separately, making things easier when calculating angular momentum square.

    Args:
        f: The function to act the angular momentum operator on. It can be:
           - Callable which evaluates the log of the magnitude of the wavefunction.
           - Any other functions. Useful to apply the operator for the second time.
        Q: Monopole strength.

    Returns:
        A function with signature
            f(params, data) -> tuple[jnp.ndarray, jnp.ndarray]
    """
    # We are not using `grad` here to make things easier when calculating $\hat L^2$,
    # where the output of `f` is a vector instead of a number. We are using reverse mode
    # jacobian since the output dimension is smaller.
    jac_f_real = jax.jacrev(lambda p, d: f(p, d).real, argnums=1)
    jac_f_imag = jax.jacrev(lambda p, d: f(p, d).imag, argnums=1)

    def angular_momentum_operator(
        params: Params, data: Float[Array, "nelec 2"]
    ) -> tuple[Complex[Array, "... 3"], Float[Array, "3"]]:
        r"""Calculate terms of angular momentum operator acting on `f`.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            A tuple of:
            - The result of the differential operator acting on `f`
                Depending on the output shape of `f`, which can be a complex number
                (usual wavefunction \log \psi) or an array of complex numbers
                ((\hat L_0 \psi) / \psi), the output shape can be different.
            - The "constant" magneric term.

            The local angular momentum is the sum of these two terms when f is logpsi.
        """
        theta, phi = data[..., 0], data[..., 1]
        jacobian = jac_f_real(params, data) + 1j * jac_f_imag(params, data)
        # Adding an extra dimension at the end for interaction with vectors
        grad_theta, grad_phi = jacobian[..., 0, None], jacobian[..., 1, None]

        r_hat = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)], axis=-1)
        theta_hat_prime = jnp.stack(  # Rescaled theta_hat with 1/sin(theta)
            [cos(phi) / tan(theta), sin(phi) / tan(theta), -jnp.ones_like(theta)],
            axis=-1,
        )

        differential_term = jnp.sum(
            -1j * (phi_hat * grad_theta - theta_hat_prime * grad_phi), axis=-2
        )
        magnetic_term = Q * jnp.sum(theta_hat_prime * cos(theta)[:, None] + r_hat, 0)

        return differential_term, magnetic_term

    return angular_momentum_operator


def make_local_angular_momentum_square(f: LogPsiNetwork, Q: float | int):
    r"""Create a function evaluating local anguar momentum square $(\hat L^2 psi)/psi$.

    Although angular momentum square operator requires second derivatives w.r.t. psi,
    it is not as expensive as Hessian. By applying $\hat L$ twice to psi, we are doing
    "2N -> 3x3" calculation instead of "2N -> 2Nx2N" as in Hessian calculation.

    However, obtaining $(\hat L^2 psi) / psi$ is not that straightforward because
    $\hat L$ has a magnetic contribution term, and the formula for local angular
    momentum square will be a little different. We write

        \hat L = \hat L_0 + C,

    where $\hat L_0$ is a pure first order differential operator, and $C$ is the
    magnetic contribution term. We first apply $\hat L$ to $\log\psi$ using function
    `make_angular_momentum_operator`, and we obtain

        (\hat L_0 \psi) / \psi, C

    Then we apply $\hat L_0$ on $(\hat L_0 \psi) / \psi + C$ again, yielding

        (\hat L_0^2 \psi) / \psi - (\hat L_0 \psi)^2 / \psi^2 + \hat L_0 C

    Comparing it with the expanded form of (\hat L^2 \psi) / \psi:

        (\hat L_0^2 \psi) / \psi + 2 C (\hat L_0 \psi) / \psi + \hat L_0 C + C^2

    The differences are

        (\hat L_0 \psi)^2 / \psi^2 + 2 C (\hat L_0 \psi) / \psi + C^2
      = ((\hat L_0 \psi) / \psi + C)^2
      = ((\hat L \psi) / \psi)^2

    Args:
        f: Callable which evaluates the log of the magnitude of the wavefunction.
        Q: Monopole strength.

    Returns:
        A function with signature
            f(params, data) -> AngularMomenta
    """
    angular_momentum_operator_on_logpsi = make_angular_momentum_operator(f, Q)
    angular_momentum_operator_square_on_logpsi = make_angular_momentum_operator(
        lambda p, d: sum(angular_momentum_operator_on_logpsi(p, d)), Q
    )

    def angular_momentum_square(
        params: Params, data: Float[Array, "nelec 2"]
    ) -> AngularMomenta:
        angular_momentum = sum(angular_momentum_operator_on_logpsi(params, data))
        angular_momentum_square_components = (
            # Dot product follow \delta_{ij} a_i b_j, and thus we only take diagonal sum
            jnp.diag(angular_momentum_operator_square_on_logpsi(params, data)[0])
            + angular_momentum**2
        )
        angular_momentum_z_square = angular_momentum_square_components[2].real
        angular_momentum_square = jnp.sum(angular_momentum_square_components).real
        return AngularMomenta(
            angular_momentum_z=angular_momentum[2].real,
            angular_momentum_square=angular_momentum_square,
            angular_momentum_z_square=angular_momentum_z_square,
        )

    return angular_momentum_square


def make_local_energy(
    f: LogPsiNetwork, system: System, laplacian_mode: LaplacianMode
) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.
        laplacian_mode: "hessian" or "forward"
            Use Hessian-based calculation of Laplacian or forward Laplacian

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux / 2
    radius = jnp.array(system.radius or jnp.sqrt(Q))
    if laplacian_mode == LaplacianMode.forward:
        ke = make_local_kinetic_energy_with_fwdlap(f, Q, radius)
    elif laplacian_mode == LaplacianMode.hessian:
        ke = make_local_kinetic_energy_with_hessian(f, Q, radius)
    pe = make_potential(system.interaction_type, Q, radius)

    def _e_l(
        params: Params, data: Float[Array, "nelec 2"]
    ) -> tuple[Complex[Array, ""], OtherObservables]:
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
