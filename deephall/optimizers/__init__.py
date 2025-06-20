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


from deephall.config import Config, OptimizerName
from deephall.loss import make_loss_fn
from deephall.types import LogPsiNetwork, TrainingInit, TrainingStep

from .adam import make_adam_training_step
from .kfac import make_kfac_training_step
from .none import make_inference_step


def make_optimizer_step(
    cfg: Config, network: LogPsiNetwork
) -> tuple[TrainingInit, TrainingStep]:
    loss_grad_fn = make_loss_fn(network, cfg.system, laplacian_mode=cfg.laplacian)
    if cfg.optim.optimizer == OptimizerName.adam:
        return make_adam_training_step(cfg.optim.adam, loss_grad_fn)
    if cfg.optim.optimizer == OptimizerName.kfac:
        return make_kfac_training_step(cfg.optim.kfac, loss_grad_fn)
    if cfg.optim.optimizer == OptimizerName.none:
        return make_inference_step(loss_grad_fn)
    raise ValueError(f"Optimizer {cfg.optim.optimizer} is not implemented!")
