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

import os

from jaxtyping import install_import_hook

cpu_count = min(os.cpu_count() or 1, 6)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count}"


modules_to_jaxtyping = ["mcmc", "hamiltonian", "loss"]
install_import_hook(
    [f"deephall.{mod}" for mod in modules_to_jaxtyping], "beartype.beartype"
)
