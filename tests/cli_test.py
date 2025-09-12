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
from click.testing import CliRunner
from omegaconf import OmegaConf

from deephall.cli import cli


@pytest.fixture
def dotlist(tmp_path: Path):
    return [
        "seed=42",
        "system.nspins=[3, 0]",
        "system.flux=6",
        "network.type=laughlin",
        "optim.iterations=100",
        "optim.optimizer=none",
        f"log.save_path={tmp_path}",
    ]


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


def test_cli(runner, dotlist: list[str]):
    result = runner.invoke(cli, ["train", *dotlist])
    assert "iterations: 100\n" in result.stderr
    assert "energy=2.58" in result.stderr
    assert "L_square=0.0000" in result.stderr


@pytest.mark.parametrize(
    "network_type",
    [
        "deephall.networks.laughlin",
        str(Path(__file__).parent.parent / "deephall" / "networks" / "laughlin.py"),
    ],
)
def test_network_type(runner, network_type: str, dotlist: list[str]):
    """Test using absolute module or file path for the network.type."""
    dotlist = [
        f"network.type={network_type}" if opt.startswith("network.type") else opt
        for opt in dotlist
    ]
    result = runner.invoke(cli, ["train", *dotlist])
    assert "L_square=0.0000" in result.stderr


def test_yml(runner, dotlist: list[str], tmp_path: Path):
    config_path = tmp_path / "config.yml"
    with config_path.open("w", encoding="utf8") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.from_dotlist(dotlist)))
    result = runner.invoke(
        cli, ["train", "--yml", str(config_path), "optim.iterations=50"]
    )

    assert "iterations: 50\n" in result.stderr
    assert "energy=2.58" in result.stderr
    assert "L_square=0.0000" in result.stderr
