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

import enum
from typing import cast

import jax
import kfac_jax
from jax import numpy as jnp
from jaxtyping import Array, Complex, Float

from deephall import constants
from deephall.config import LaplacianMode, System
from deephall.hamiltonian import OtherObservables, make_local_energy
from deephall.types import LogPsiNetwork, LossStats, Params


def iqr_clip_real(x: Float[Array, " n"], scale=100.0) -> Float[Array, " n"]:
    """Clip the observables based on interquartile range (IQR)."""
    q1 = jnp.nanquantile(x, 0.25)
    q3 = jnp.nanquantile(x, 0.75)
    iqr = q3 - q1
    return jnp.clip(x, q1 - scale * iqr, q3 + scale * iqr)


def iqr_clip(x: Complex[Array, " n"], scale=100.0) -> Complex[Array, " n"]:
    """Clip complex observables by applying IQR clip on both real and imag parts."""
    return iqr_clip_real(x.real, scale) + 1j * iqr_clip_real(x.imag, scale)


class LossMode(enum.Enum):
    ENERGY_GRAD = enum.auto()
    ENERGY_DIFF = enum.auto()
    SR_F_VECTOR = enum.auto()


def make_loss_fn(
    network: LogPsiNetwork,
    system: System,
    mode: LossMode = LossMode.ENERGY_GRAD,
    laplacian_mode: LaplacianMode = LaplacianMode.hessian,
):
    r"""Create the loss function and its gradient for the neural network.

    The loss function is just the sum of the (clipped) average energy and other penalty
    term. The corresponding gradient with respect to the parameters is evaluated with:

    \frac{\partial}{\partial\alpha}\langle O\rangle = 2\Re[
        \langle O\frac{\partial}{\partial\alpha}\log\psi\rangle
        -\langle O\rangle \langle \frac{\partial}{\partial\alpha}\log\psi\rangle
    ]
    """
    loss_fn = make_local_energy(network, system, laplacian_mode=laplacian_mode)
    batch_local_energy = jax.vmap(loss_fn, in_axes=(None, 0))

    df_real = jax.vmap(
        jax.grad(lambda params, x: network(params, x).real), in_axes=(None, 0)
    )
    df_imag = jax.vmap(
        jax.grad(lambda params, x: network(params, x).imag), in_axes=(None, 0)
    )
    batch_network = jax.vmap(network, in_axes=(None, 0))

    def loss_prod(grad_logpsi_conj, diff):
        diff = diff.reshape(
            diff.shape + (1,) * (len(grad_logpsi_conj.shape) - len(diff.shape))
        )
        return jnp.nan_to_num(2 * jnp.nanmean(grad_logpsi_conj * diff, axis=0))

    def loss_and_grad(
        params: Params, data: Float[Array, "batch nelec 2"]
    ) -> tuple[LossStats, Params | Complex[Array, " batch"]]:
        el, other_observables = batch_local_energy(params, data)
        pmean_observables = cast(
            OtherObservables,
            jax.tree.map(lambda x: constants.pmean(jnp.mean(x)), other_observables),
        )

        loss = constants.pmean(jnp.nanmean(el))
        clipped_loss = constants.pmean(jnp.nanmean(iqr_clip(el)))
        diff_to_clip = el - clipped_loss
        if system.lz_penalty:
            lz_square = other_observables["angular_momentum_z_square"]
            lz = other_observables["angular_momentum_z"]
            clipped_lz_square = constants.pmean(jnp.nanmean(iqr_clip(lz_square)))
            clipped_lz = constants.pmean(jnp.nanmean(iqr_clip(lz)))
            diff_to_clip += system.lz_penalty * (
                (lz_square - clipped_lz_square)
                - 2 * system.lz_center * (lz - clipped_lz)
            )
        if system.l2_penalty:
            l2 = other_observables["angular_momentum_square"]
            clipped_l2 = constants.pmean(jnp.nanmean(iqr_clip(l2)))
            diff_to_clip += system.l2_penalty * (l2 - clipped_l2)
        diff = iqr_clip(diff_to_clip)

        variance = constants.pmean(jnp.nanmean(el.real**2) - loss.real**2)
        stats = LossStats(**pmean_observables, energy=loss, variance=variance)
        if mode == LossMode.ENERGY_DIFF:
            return stats, diff

        kfac_jax.register_squared_error_loss(batch_network(params, data)[:, None])
        tangent_out = jax.tree.map(
            lambda real, imag: loss_prod(real - 1j * imag, diff),
            df_real(params, data),
            df_imag(params, data),
        )

        if mode == LossMode.ENERGY_GRAD:
            return stats, jax.tree.map(jnp.real, tangent_out)
        elif mode == LossMode.SR_F_VECTOR:
            return stats, tangent_out

    return loss_and_grad
