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

import click

from deephall.cli.train import train


@click.group(
    help="Simulating the fractional quantum Hall effect (FQHE) with "
    "neural network variational Monte Carlo.\n\n"
    "To start training, use `deephall train` instead of `deephall`.",
)
@click.pass_context
def cli(ctx: click.Context):
    pass


cli.add_command(train)
