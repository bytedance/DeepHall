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

from typing import cast

import kfac_jax
from chex import PRNGKey
from jax import numpy as jnp

from deephall import constants
from deephall.config import OptimizerKfac
from deephall.log import CheckpointState
from deephall.loss import LossStats
from deephall.types import TrainingInit, TrainingStep

from .complex_support import PatchedBlockDiagonalCurvature
from .curvature_tags_and_blocks import GRAPH_PATTERNS


def make_kfac_training_step(
    optim_cfg: OptimizerKfac, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:
    def val_and_grad(params, data):
        stats, grads = loss_grad_fn(params, data)
        return (stats["energy"], stats), grads

    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        custom_estimator_ctor=PatchedBlockDiagonalCurvature,  # support complex
        num_burnin_steps=0,  # burn in requires data iterator, which is not implemented
        value_func_has_aux=True,  # LossStats are returned as aux data
        multi_device=True,  # automatically uses pmap
        l2_reg=optim_cfg.l2_reg,
        norm_constraint=optim_cfg.norm_constraint,
        learning_rate_schedule=optim_cfg.lr.schedule,
        curvature_ema=optim_cfg.curvature_ema,
        inverse_update_period=optim_cfg.inverse_update_period,
        estimation_mode="fisher_exact",
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(graph_patterns=GRAPH_PATTERNS),
    )
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(optim_cfg.damping)
    )

    def init(params, key, data):
        return optimizer.init(params, key, data)

    def step(state: CheckpointState, key: PRNGKey):
        params, data, opt_state, mcmc_width = state
        params, opt_state, *_, stats = optimizer.step(
            params=params,
            state=opt_state,
            rng=key,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )
        return (
            CheckpointState(params, data, opt_state, mcmc_width),
            cast(LossStats, stats["aux"]),
        )

    return init, step
