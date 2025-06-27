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

from pathlib import Path

import pytest
from pytest import CaptureFixture

from deephall import Config, train
from deephall.config import OptimizerName


@pytest.fixture
def simple_config():
    config = Config()
    config.seed = 42
    config.system.nspins = (3, 0)
    config.system.flux = 2
    config.system.interaction_strength = 0.0
    config.optim.iterations = 100
    config.network.psiformer.num_layers = 1
    config.network.psiformer.num_heads = 1
    config.network.psiformer.heads_dim = 4
    config.batch_size = 60
    config.log.initial_energy = False
    return config


def test_training(simple_config: Config, tmp_path: Path, capsys: CaptureFixture[str]):
    simple_config.log.save_path = str(tmp_path)
    train(simple_config)
    assert (tmp_path / "train_stats.csv").exists()
    assert (tmp_path / "ckpt_000099.npz").exists()

    captured = capsys.readouterr()
    # Energy is oscillating around 1.5
    assert "energy=1.5" in captured.err
    assert "energy=1.4" in captured.err


@pytest.mark.parametrize("optimizer", (OptimizerName.kfac, OptimizerName.adam))
def test_checkpoint(
    simple_config: Config,
    tmp_path: Path,
    capsys: CaptureFixture[str],
    optimizer: OptimizerName,
):
    simple_config.optim.iterations = 1
    simple_config.optim.optimizer = optimizer
    simple_config.log.save_path = str(tmp_path)
    train(simple_config)
    assert (tmp_path / "ckpt_000000.npz").exists()

    simple_config.optim.iterations = 2
    train(simple_config)
    assert (tmp_path / "ckpt_000001.npz").exists()

    captured = capsys.readouterr()
    assert "Restored checkpoint" in captured.err
