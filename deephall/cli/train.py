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

import chex
import click
from omegaconf import OmegaConf

from deephall.config import Config
from deephall.train import train as run_train


@click.command(
    name="train",
    help="Start training networks.\n\n"
    "DOTLIST is `path.to.key=value` pairs used for specifiying configuration.",
)
@click.argument("dotlist", nargs=-1, required=False)
@click.option("--yml", help="config YML file to merge", type=click.Path(exists=True))
@click.option("--debug", help="disable JAX pmap", is_flag=True)
def train(dotlist: tuple[str, ...], yml: str | None, debug: bool) -> None:
    # Show help if no arguments provided (mimicking the original behavior)
    if not dotlist and not yml and not debug:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    config = OmegaConf.structured(Config)
    if yml:
        config = OmegaConf.merge(config, OmegaConf.load(yml))
    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(dotlist)))

    if debug:
        with chex.fake_pmap_and_jit():
            run_train(Config.from_dict(cast(dict, config)))
    else:
        run_train(Config.from_dict(cast(dict, config)))


if __name__ == "__main__":
    train()
