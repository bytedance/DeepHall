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

import click
import jax
import numpy as np
from chex import ArrayTree
from upath import UPath

from deephall.types import CheckpointState


def pytree_as_npz(name: str, tree: ArrayTree) -> dict[str, np.ndarray]:
    """Save PyTree to npz.

    Args:
        name: The name of the array in npz file.
        tree: PyTree to be saved.

    Returns:
        dict of file name and arrays to be used by `np.savez`.
    """
    vals, _ = jax.tree.flatten(tree)
    return {f"{name}/{i}": val for i, val in enumerate(vals)}


def convert_single_checkpoint(ckpt: str | Path | UPath) -> None:
    """Convert a given checkpoint to the new version."""
    ckpt_path = UPath(ckpt)
    with ckpt_path.open("rb") as npf, np.load(npf, allow_pickle=True) as f:
        if "params/0" in f:
            raise ValueError(f"Checkpoint {ckpt} is already in the new format.")
        step = f["step"].tolist()
        state = CheckpointState(
            f["params"].tolist(), f["data"], f["opt_state"].tolist(), f["mcmc_width"]
        )
    backup_path = ckpt_path.with_suffix(".npz.bak")
    ckpt_path.rename(backup_path)
    try:
        if hasattr(state.opt_state, "estimator_state"):  # assume KFAC optimizer
            # The implementation of `WeightedMovingAverage` changed in kfac-jax v0.0.7
            # See https://github.com/google-deepmind/kfac-jax/commit/baaec40
            for block_state in state.opt_state.estimator_state.blocks_states:
                factors = getattr(
                    block_state, "diagonal_factors", getattr(block_state, "factors", [])
                )
                for factor in factors:
                    factor.value = jax.tree_util.tree_map(
                        lambda x: x / factor.weight, factor.raw_value
                    )
        with ckpt_path.open("wb") as f:
            np.savez_compressed(
                f,
                allow_pickle=False,
                step=step,
                data=state.data,
                mcmc_width=state.mcmc_width,
                **pytree_as_npz("params", state.params),
                **pytree_as_npz("opt_state", state.opt_state),
            )
    except Exception as e:
        backup_path.rename(ckpt_path)
        raise e


@click.command("ckpt-conv", help="Convert checkpoints to new version.")
@click.argument("ckpt")
@click.option("--recursive", "-r", is_flag=True)
def convert_checkpoint(ckpt: str, recursive=False):
    ckpt_path = UPath(ckpt)
    if not ckpt_path.exists():
        raise click.BadParameter(f"{ckpt} does not exist.")
    if ckpt_path.is_dir():
        if not recursive:
            raise click.BadParameter("Must enable recusive flag to handle directories.")
        for individual_ckpt_path in ckpt_path.rglob("ckpt_[0-9]*.npz"):
            convert_single_checkpoint(individual_ckpt_path)
    else:
        convert_single_checkpoint(ckpt_path)


if __name__ == "__main__":
    convert_checkpoint()
