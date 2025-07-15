# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Ltd. and/or its affiliates
# ("Bytedance's Modifications"). All Bytedance's Modifications are
# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates.

"""Define and register custom KFAC blocks.

The default handling of dense blocks are not flexible enough, and thus we need to
handle it by ourselves. Besides, as elaborated in `complex_support.py`, we should
update the estimation of curvatures to support complex numbers.
"""

from collections.abc import Sequence
from functools import partial
from math import prod
from string import ascii_lowercase

import jax
import kfac_jax
from jax import numpy as jnp


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap).

    By default, kfac_jax will assume that the blocks only transforms the last axis,
    i.e. `ij,jk->ik`. However, in general, the dense block can be repeatedly applied
    (i.e. transforming the last axis and keeping all other axis onchanged), or we can
    transform more axis, and the output shape is not necessarily the same as the input.
    Therefore, we need to modify the ways to handle the parameters.
    """

    def __init__(self, layer_tag_eq):
        # Even though the superclass constructor will set this later, we need to do
        # it now since it's used below by `self.parameters_shapes`.
        self._layer_tag_eq = layer_tag_eq

        parameters_specs = []
        _, w_dim_out = self.weight_dimension_split

        for shape in self.parameters_shapes:
            in_str = ascii_lowercase[: len(shape)]
            # Don't use `w_dim_in` because it's not applicable to bias
            out_str = f"({in_str[:-w_dim_out]})({in_str[-w_dim_out:]})"
            parameters_specs.append(f"{in_str} -> {out_str}")

        super().__init__(layer_tag_eq, parameters_specs)

    @property
    def weight_dimension_split(self) -> tuple[int, int]:
        """Determine the input/output dimensions transformed by the weight matrix.

        The formula derives these values based on the input/output tensor dimensions
        and the weight's shape to ensure proper contraction during matrix operations.

        Returns:
            - w_dim_in: Dimensions of the input's feature space transformed by weight.
            - w_dim_out: Dimensions of the output's transformed feature space.
        """
        input_dim = len(self.inputs_shapes[0])
        output_dim = len(self.outputs_shapes[0])
        w_dim = len(self.parameters_shapes[0])
        # No further assumption. It is equired for proper contraction
        w_dim_in = (w_dim - (output_dim - input_dim)) // 2
        w_dim_out = w_dim - w_dim_in
        return w_dim_in, w_dim_out

    def fixed_scale(self) -> kfac_jax.utils.Numeric:
        """A fixed scalar pre-factor of the curvature (e.g. constant)."""
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.KroneckerFactored.State,  # type: ignore
        estimation_data: kfac_jax.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.KroneckerFactored.State:
        """Reshape the imputs and outputs take care of the complex conjugate."""
        assert 1 <= self.number_of_parameters <= 2

        # Copy this first since we mutate it later in this function.
        state = state.copy()

        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        assert x.shape[0] == batch_size

        w_dim_in, w_dim_out = self.weight_dimension_split
        feature_size_in = prod(x.shape[-w_dim_in:])
        feature_size_out = prod(dy.shape[-w_dim_out:])

        x = x.reshape([-1, feature_size_in])
        dy = dy.reshape([-1, feature_size_out])
        batch_size = x.size // feature_size_in
        assert all(arg.shape[0] == batch_size for arg in (x, dy))

        if self.number_of_parameters == 2:
            x_one = jnp.ones_like(x[:, :1])
            x = jnp.concatenate([x, x_one], axis=1)

        input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
        output_stats = jnp.einsum("ay,az->yz", dy.conj(), dy).real / batch_size

        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)

        return state


class NaiveDiagonal(kfac_jax.Diagonal):
    """Approximates the diagonal of the curvature with in the most obvious way."""

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.Diagonal.State,  # type: ignore
        estimation_data: kfac_jax.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.Diagonal.State:
        del identity_weight
        state = state.copy()

        for factor, dw in zip(state.diagonal_factors, estimation_data.tangents.params):
            # Take care of complex numbers
            factor.update(jnp.real(dw.conj() * dw) / batch_size, ema_old, ema_new)

        return state


def _repeated_dense(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    """Example of a dense layer function."""
    w, *opt_b = params
    y = jax.lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())))
    if opt_b:
        b = opt_b[0]
        y += b.reshape((1, *b.shape))
    return y


def _repeated_dense_attention_out(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    """Example of a dense layer function."""
    w, b = params
    y = jax.lax.dot_general(x, w, (((x.ndim - 2, x.ndim - 1), (0, 1)), ((), ())))
    y += b.reshape((1, *b.shape))
    return y


def _repeated_dense_complex_no_bias(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    [w] = params
    w = w.astype(jnp.complex64)
    y = jax.lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())))
    return y


_dense_parameter_extractor = partial(
    kfac_jax.tag_graph_matcher._dense_parameter_extractor,  # pylint: disable=protected-access
    variant="repeated_dense",
)


GRAPH_PATTERNS = (
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_with_bias",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4]), [jnp.zeros([4, 3]), jnp.zeros([3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_no_bias",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4]), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_more_dim",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([4, 3]), jnp.zeros(3)]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_more_dim_no_bias",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_complex_no_bias",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense_complex_no_bias,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4], dtype=jnp.complex64), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_attention_with_bias",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=_repeated_dense_attention_out,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([3, 4, 3]), jnp.zeros([3])]],
    ),
    *kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS,
)

kfac_jax.set_default_tag_to_block_ctor("repeated_dense", RepeatedDenseBlock)
kfac_jax.set_default_tag_to_block_ctor("generic", NaiveDiagonal)
