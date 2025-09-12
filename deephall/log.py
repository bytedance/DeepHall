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

import datetime
import difflib
import logging
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import jax
import numpy as np
from chex import ArrayTree
from jax import numpy as jnp
from omegaconf import OmegaConf
from upath import UPath

from deephall.config import Config
from deephall.types import CheckpointState

logger = logging.getLogger("deephall")


def dedup_pytree_as_npz(name: str, tree: ArrayTree) -> dict[str, jnp.ndarray]:
    """Save PyTree to npz and remove duplication in devices.

    Args:
        name: The name of the array in npz file.
        tree: PyTree to be saved.

    Returns:
        dict of file name and arrays to be used by `np.savez`.
    """
    vals, _ = jax.tree.flatten(tree)
    return {f"{name}/{i}": val[0] for i, val in enumerate(vals)}


def redup_npz_as_pytree(
    name: str, npf: dict, ref_pytree: ArrayTree, cards: int = 1
) -> ArrayTree:
    """Restore PyTree from npz and add duplication across devices.

    Args:
        name: The name of the array in npz file.
        npf: Opened NumPy file object.
        ref_pytree: Reference PyTree whose structure will be used for restore.
        cards: Number of duplications to be made.

    Returns:
        The restored PyTree.
    """
    _, treedef = jax.tree.flatten(ref_pytree)
    vals = [
        jnp.repeat(npf[f"{name}/{i}"][None], cards, 0)
        for i in range(treedef.num_leaves)
    ]
    return jax.tree.unflatten(treedef, vals)


def init_logging():
    """Set up Python logging and absl logging (for JAX)."""
    from absl import logging as absl_logging

    # kfac_jax uses absl for logging so we need to configure it
    absl_logging.set_verbosity(absl_logging.INFO)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    # Avoid pollution from absl
    logger.propagate = False


class StatsWriter:
    """Simple file wrap supporting CSV headers and force flush."""

    def __init__(self, stats_path: UPath):
        self.stats_path = stats_path
        self.stats_file = None
        self.hidden_fields: set[str] = set()

    def __enter__(self):
        self.should_write_head = (
            # If there's not CSV file or the file is empty
            not self.stats_path.exists() or self.stats_path.stat().st_size == 0
        )
        mode = "a" if self.stats_path.exists() else "w"
        self.stats_file = self.stats_path.open(mode, buffering=1)
        self.stats_file.__enter__()
        return self

    def hide(self, *args):
        """Hide these fields in the stderr but still show them in CSV."""
        self.hidden_fields.update(args)

    def log(self, **kwargs):
        """Write the key-value pairs to CSV and stderr."""
        if self.should_write_head:
            self.stats_file.write(",".join(kwargs.keys()) + "\n")
            self.should_write_head = False
        self.stats_file.write(",".join(kwargs.values()) + "\n")
        info_to_print = ", ".join(
            f"{k}={v}" for k, v in kwargs.items() if k not in self.hidden_fields
        )
        logger.info(info_to_print)

    def force_flush(self):
        """Force flush the file.

        Some file systems does not have reliable flush. A force flush closes
        the file and then opens it, forcing the content update.
        """
        self.stats_file.__exit__(None, None, None)
        self.stats_file = self.stats_path.open("a", buffering=1)
        self.stats_file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stats_file.__exit__(exc_type, exc_value, traceback)
        if self.should_write_head:
            # Remove file if nothing got written
            self.stats_path.unlink(missing_ok=True)


class LogManager:
    def __init__(self, cfg: Config):
        if cfg.log.save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = UPath(
                f"DeepHall_n{sum(cfg.system.nspins)}l{cfg.system.flux}_{timestamp}"
            )
        else:
            self.save_path = UPath(cfg.log.save_path)

        if cfg.log.restore_path is None:
            self.restore_path = self.save_path
        else:
            self.restore_path = UPath(cfg.log.restore_path)
            if not self.restore_path.exists():
                logger.warning("Restore path %s does not exist!", self.restore_path)

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        self.check_config(cfg)

    def check_config(self, cfg: Config) -> None:
        """Save the current config and compare with the previous one if available."""
        restore_config_path = self.restore_path / "config.yml"
        current_config_yaml = [f"git_commit: {get_git_commit()}\n"]
        # keepends is required for difflib
        current_config_yaml.extend(OmegaConf.to_yaml(cfg).splitlines(keepends=True))

        if restore_config_path.exists():
            with restore_config_path.open() as f:
                original_config_yaml = f.readlines()
        else:
            original_config_yaml = []
        sys.stderr.writelines(difflib.ndiff(original_config_yaml, current_config_yaml))
        with (self.save_path / "config.yml").open("w") as f:
            f.writelines(current_config_yaml)

    def save_checkpoint(self, step: int, state: CheckpointState) -> None:
        assert state.data.ndim == 4, "data has wrong shape"

        ckpt_path = self.save_path / f"ckpt_{step:06d}.npz"
        logger.info("Saving checkpoint %s", ckpt_path)

        with ckpt_path.open("wb") as f:
            np.savez_compressed(
                f,
                allow_pickle=False,
                step=step,
                data=state.data.reshape(-1, *state.data.shape[2:]),
                mcmc_width=state.mcmc_width[0],
                **dedup_pytree_as_npz("params", state.params),
                **dedup_pytree_as_npz("opt_state", state.opt_state),
            )

    def try_restore_checkpoint(
        self, fallback: tuple[int, CheckpointState]
    ) -> tuple[int, CheckpointState]:
        """Try to restore checkpoints from `restore_path`."""
        template_state = fallback[1]
        if not self.restore_path.exists():
            return fallback
        if self.restore_path.is_file():
            return self.restore_checkpoint(
                self.restore_path, template_state.params, template_state.opt_state
            )
        for ckpt_path in sorted(self.restore_path.glob("ckpt_*.npz"), reverse=True):
            ckpt_path = cast(UPath, ckpt_path)
            try:
                return self.restore_checkpoint(
                    ckpt_path, template_state.params, template_state.opt_state
                )
            except Exception as e:
                logger.warning("Error restoring checkpoint %s: %s", ckpt_path, e)
        return fallback

    @staticmethod
    def restore_checkpoint(
        ckpt: str | Path | UPath,
        template_params: ArrayTree,
        template_opt_state: ArrayTree,
    ) -> tuple[int, CheckpointState]:
        """Resore a given checkpoint.

        Args:
            ckpt: Checkpoint path.
            template_params: Reference object whose structure is used to restore
                parameters.
            template_opt_state: Reference object whose structure is used to restore
                opt_state.

        Returns:
            A tuple containing current step and state.
        """
        ckpt_path = UPath(ckpt)
        with ckpt_path.open("rb") as npf, np.load(npf) as f:
            step = f["step"].tolist() + 1
            data = f["data"]
            assert data.ndim == 3, "data has wrong shape"

            cards = jax.local_device_count()
            state = CheckpointState(
                redup_npz_as_pytree("params", f, template_params, cards),
                data.reshape(cards, -1, *data.shape[1:]),
                redup_npz_as_pytree("opt_state", f, template_opt_state, cards),
                jnp.ones(cards) * f["mcmc_width"],
            )
            logger.info("Restored checkpoint %s", ckpt_path)
            return step, state

    @contextmanager
    def create_writer(self) -> Generator[StatsWriter, None, None]:
        """Creates a writer with `writer.log` to conveniently process the logs."""
        with StatsWriter(cast(UPath, self.save_path / "train_stats.csv")) as writer:
            yield writer


def get_git_commit():
    """Get current git revision if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "''"
