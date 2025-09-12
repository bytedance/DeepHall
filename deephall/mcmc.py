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
import numpy as np
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float

from deephall import constants
from deephall.types import LogPsiNetwork, Params


def mh_update(
    params: Params,
    f: LogPsiNetwork,
    x1: Float[Array, "batch nelec 2"],
    key: PRNGKey,
    lp_1: Float[Array, " batch"],
    num_accepts: Float[Array, ""],
    stddev: float | Float[Array, ""] = 0.02,
):
    """Performs one Metropolis-Hastings step using an all-electron move.

    Args:
      params: Wavefuncttion parameters.
      f: Callable with signature f(params, x) which returns the log of the
        wavefunction (i.e. the sqaure root of the log probability of x).
      x1: Initial MCMC configurations.
      key: RNG state.
      lp_1: log probability of f evaluated at x1 given parameters params.
      num_accepts: Number of MH move proposals accepted.
      stddev: width of Gaussian move proposal.

    Returns:
      (x, key, lp, num_accepts), where:
        x: Updated MCMC configurations.
        key: RNG state.
        lp: log probability of f evaluated at x.
        num_accepts: update running total of number of accepted MH moves.
    """
    key_new, key_sample, key_cond = jax.random.split(key, 3)
    x2 = sph_sampling(key_sample, x1, stddev)
    lp_2 = 2.0 * f(params, x2).real  # log prob of proposal
    ratio = lp_2 - lp_1

    rnd = jnp.log(jax.random.uniform(key_cond, shape=lp_1.shape))
    cond = ratio > rnd
    x_new = jnp.where(cond[..., None, None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)

    return x_new, key_new, lp_new, num_accepts


def sph_sampling(
    key: PRNGKey, x1: Float[Array, "batch nelec 2"], stddev: float | Float[Array, ""]
) -> Float[Array, "batch nelec 2"]:
    """Propose electrons MCMC move on the sphere.

    Suppose the electron is at the north pole, similar to the Euclidean Gaussian MCMC
    proposal, the proposal on the sphere also obeys Gaussian distribution in theta and
    is symmetric in phi. Of course the real electrons are not at the north pole, but we
    can always perform a change of coordinates to place the electron of interest to the
    north pole, generate the proposals in the transformed coordinates (theta'/phi'),
    and then transform them back to the original coordinates.

    Args:
        key: Random key
        x1: Current electrons configuration.
        stddev: The Gaussian width of the move.

    Returns:
        New electron configuration.
    """
    theta, phi = x1[..., 0], x1[..., 1]
    key_theta, key_phi = jax.random.split(key)
    # Assuming the electrons are on the north pole, and work on theta' - phi' coord
    theta_prime = jnp.arctan(jax.random.normal(key_theta, shape=theta.shape) * stddev)
    phi_prime = jax.random.uniform(key_phi, phi.shape) * 2 * jnp.pi
    xyz_prime: Float[Array, "batch nelec 3"] = jnp.stack(
        [
            jnp.sin(theta_prime) * jnp.cos(phi_prime),
            jnp.sin(theta_prime) * jnp.sin(phi_prime),
            jnp.cos(theta_prime),
        ],
        axis=-1,
    )
    one = jnp.ones_like(phi)
    zero = jnp.zeros_like(phi)
    # We then rotate the pole pointing to the direction of each electron
    rot_z: Float[Array, "3 3 batch nelec"] = jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), zero],
            [jnp.sin(phi), jnp.cos(phi), zero],
            [zero, zero, one],
        ]
    )
    rot_y: Float[Array, "3 3 batch nelec"] = jnp.array(
        [
            [jnp.cos(theta), zero, jnp.sin(theta)],
            [zero, one, zero],
            [-jnp.sin(theta), zero, jnp.cos(theta)],
        ]
    )
    x2_xyz = jnp.einsum("ijbn,jkbn,bnk->bni", rot_z, rot_y, xyz_prime)
    x2, y2, z2 = x2_xyz[..., 0], x2_xyz[..., 1], x2_xyz[..., 2]
    theta = jnp.arccos(jnp.clip(z2, -1, 1))
    phi = jnp.sign(y2) * jnp.arccos(jnp.clip(x2 / jnp.sin(theta), -1, 1))
    return jnp.stack([theta, phi], axis=-1)


def make_mcmc_step(
    batch_network: LogPsiNetwork, batch_per_device: int, steps: int = 10
):
    """Creates the MCMC step function.

    Args:
      batch_network: function, signature (params, x), which evaluates the log of
        the wavefunction (square root of the log probability distribution) at x
        given params. Inputs and outputs are batched.
      batch_per_device: Batch size per device.
      steps: Number of MCMC moves to attempt in a single call to the MCMC step
        function.

    Returns:
      Callable which performs the set of MCMC steps.
    """

    @jax.jit
    def mcmc_step(
        params: Params,
        data: Float[Array, "batch nelec 2"],
        key: PRNGKey,
        width: float | Float[Array, ""],
    ):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          data: (batched) MCMC configurations to pass to the network.
          key: RNG state.
          width: standard deviation to use in the move proposal.

        Returns:
          (data, pmove), where data is the updated MCMC configurations, key the
          updated RNG state and pmove the average probability a move was accepted.
        """

        def step_fn(_, x):
            return mh_update(params, batch_network, *x, stddev=width)

        logprob = 2.0 * batch_network(params, data).real
        data, key, _, num_accepts = lax.fori_loop(
            0, steps, step_fn, (data, key, logprob, 0.0)
        )
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return data, pmove

    return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Updates the width in MCMC steps.

    Args:
      t: Current step.
      width: Current MCMC width.
      adapt_frequency: The number of iterations after which the update is applied.
      pmove: Acceptance ratio in the last step.
      pmoves: Acceptance ratio over the last N steps, where N is the number of
        steps between MCMC width updates.
      pmove_max: The upper threshold for the range of allowed pmove values
      pmove_min: The lower threshold for the range of allowed pmove values

    Returns:
      width: Updated MCMC width.
      pmoves: Updated `pmoves`.
    """
    t_since_mcmc_update = t % adapt_frequency
    # update `pmoves`; `pmove` should be the same across devices
    pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
    if t > 0 and t_since_mcmc_update == 0:
        if np.mean(pmoves) > pmove_max:
            width *= 1.1
        elif np.mean(pmoves) < pmove_min:
            width /= 1.1
    return width, pmoves
