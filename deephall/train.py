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

import logging
import signal
import sys
import time
from argparse import ArgumentParser
from typing import cast

import chex
import jax
import kfac_jax
import numpy as np
from chex import PRNGKey
from flax import linen as nn
from jax import numpy as jnp
from omegaconf import OmegaConf

from deephall import constants, mcmc, optimizers
from deephall.config import Config, OptimizerName
from deephall.log import CheckpointState, LogManager, init_logging
from deephall.loss import LossMode, make_loss_fn
from deephall.networks import make_network
from deephall.types import LogPsiNetwork

logger = logging.getLogger("deephall")


def init_guess(key: PRNGKey, batch: int, nelec: int):
    """Create uniform samples on the sphere.

    Args:
        key: random key.
        batch: number of samples to generate.
        nelec: number of electrons.

    Returns:
        Electron coordinates of shape [batch, nelec, 2]
    """
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def initialize_state(cfg: Config, model: nn.Module):
    key_data, key_params = jax.random.split(jax.random.PRNGKey(cfg.seed))
    data = init_guess(key_data, cfg.batch_size, sum(cfg.system.nspins))
    data = data.reshape((jax.device_count(), -1, *data.shape[-2:]))
    params = kfac_jax.utils.replicate_all_local_devices(
        model.init(key_params, data[0, 0])
    )
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(cfg.mcmc.width))
    return 0, CheckpointState(params, data, None, mcmc_width)


def setup_mcmc(cfg: Config, network: LogPsiNetwork):
    batch_network = jax.vmap(network, in_axes=(None, 0))
    mcmc_step = mcmc.make_mcmc_step(
        batch_network,
        batch_per_device=cfg.batch_size // jax.device_count(),
        steps=cfg.mcmc.steps,
    )
    pmap_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    return pmap_mcmc_step, pmoves


def train_loop(cfg: Config, log_manager: LogManager):
    model = make_network(cfg.system, cfg.network)
    network = cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmoves = setup_mcmc(cfg, network)
    opt_init, training_step = optimizers.make_optimizer_step(cfg, network)

    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    initial_step, (params, data, opt_state, mcmc_width) = (
        log_manager.try_restore_checkpoint(fallback=initialize_state(cfg, model))
    )

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if cfg.optim.reset or opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        opt_state = opt_init(params, subkey, data)

    logger.info("Start VMC with %s JAX devices", jax.device_count())

    if initial_step == 0:
        for _ in range(cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            data, pmove = pmap_mcmc_step(params, data, subkey, mcmc_width)
        logger.info("Burn in MCMC complete")
        if cfg.log.initial_energy:
            # Logging inital energy is helpful for debugging. If we have initial energy
            # but have error in training, it's probably optimizer's fault
            initial_stats, _ = constants.pmap(
                make_loss_fn(
                    network,
                    cfg.system,
                    LossMode.ENERGY_DIFF,
                    laplacian_mode=cfg.laplacian,
                )
            )(params, data)
            logger.info("Initial energy: %s", initial_stats["energy"][0].real)

    state = CheckpointState(params, data, opt_state, mcmc_width)

    for step in range(initial_step, cfg.optim.iterations):
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        new_data, pmove = pmap_mcmc_step(
            state.params, state.data, subkey, state.mcmc_width
        )
        new_mcmc_width, pmoves = mcmc.update_mcmc_width(
            step - initial_step,
            state.mcmc_width,
            cfg.mcmc.adapt_frequency,
            pmove,
            pmoves,
        )
        state = state._replace(data=new_data, mcmc_width=new_mcmc_width)
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state, stats = training_step(state, subkey)
        yield step, state, stats, pmove


def train(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    time_start = None  # exclude jit warm up step
    steps = 0
    last_save_time = time.time()
    killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        for step, state, stats, pmove in train_loop(cfg, log_manager):
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                energy=f"{stats['energy'].real[0]:.4f}",
                energy_imag=f"{stats['energy'].imag[0]:+.4f}",
                potential=f"{stats['potential'][0]:.4f}",
                kinetic=f"{stats['kinetic'].real[0]:.4f}",
                variance=f"{stats['variance'][0]:.4f}",
                Lz=f"{stats['angular_momentum_z'][0]:+.4f}",
                Lz_square=f"{stats['angular_momentum_z_square'][0]:.4f}",
                L_square=f"{stats['angular_momentum_square'][0]:.4f}",
            )

            current_time = time.time()
            if time_start is None:
                time_start = current_time
            else:
                steps += 1

            if (
                jnp.isnan(stats["energy"].real).any()
                or step == cfg.optim.iterations - 1
                or killer.kill_now
                or (
                    current_time - last_save_time > cfg.log.save_time_interval
                    and (step + 1) % cfg.log.save_step_interval == 0
                )
            ):
                last_save_time = current_time
                writer.force_flush()
                log_manager.save_checkpoint(step, state)
            if killer.kill_now or jnp.isnan(stats["energy"].real).any():
                raise SystemExit("=" * 30 + " ABORT " + "=" * 30)
    if steps > 0 and time_start is not None:
        logger.info("Time per step: %.3fs", (current_time - time_start) / steps)


class GracefulKiller:
    """Capture SIGINT and SIGTERM so that we can save checkpoints before exit."""

    kill_now = False

    def __init__(self):
        self.original_int = signal.signal(signal.SIGINT, self.exit_gracefully)
        self.original_term = signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Mark as exit and restore signal handlers."""
        del signum, frame
        if self.kill_now:  # Only handle the first signal
            return
        print("\r", end="")  # Clear ^C
        signal.signal(signal.SIGINT, self.original_int)
        signal.signal(signal.SIGTERM, self.original_term)
        self.kill_now = True


def cli(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        prog="deephall",
        description="Simulating the fractional quantum Hall effect (FQHE) with "
        "neural network variational Monte Carlo.",
    )
    parser.add_argument(
        "dotlist", help="path.to.key=value pairs for configuration", nargs="*"
    )
    parser.add_argument("--yml", help="config YML file to merge")
    parser.add_argument("--debug", help="disable JAX pmap", action="store_true")
    args = parser.parse_args(argv or sys.argv[1:] or ["--help"])

    config = OmegaConf.structured(Config)
    if args.yml:
        config = OmegaConf.merge(config, OmegaConf.load(args.yml))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.dotlist))
    if args.debug:
        with chex.fake_pmap_and_jit():
            train(Config.from_dict(cast(dict, config)))
    else:
        train(Config.from_dict(cast(dict, config)))


if __name__ == "__main__":
    cli()
