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

import importlib.util
import sys
import uuid
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from flax import linen as nn

from deephall.config import System


def import_module_or_file(module_name: str) -> Any:
    """Import a python module or a python file.

    Args:
        module_name: the name of the module or file.
            If it ends with ".py", it will be considered as a file, otherwise module.

    Returns:
        Contents of the module.

    Raises:
        OSError: Python ifle not found.
    """
    if module_name.endswith(".py"):
        # generate unique module name
        module_id = "netobs_" + str(uuid.uuid4()).replace("-", "_")
        # `imp` is deprecated. Using `importlib` way
        spec = importlib.util.spec_from_file_location(module_id, module_name)
        if spec is None or spec.loader is None:
            raise OSError(f"Failed to load {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = module
        spec.loader.exec_module(module)
        return module
    try:
        return import_module("." + module_name, __name__)
    except ModuleNotFoundError:
        return import_module(module_name)


def resolve_object(name: str) -> Any:
    """Resolve object and option from "module:name" natation.

    Supported notations:
    - "module": resolve default object
    - "module:name": resolve `module.name`
    """
    colon_count = name.count(":")
    if colon_count == 0:
        module, obj_name = name, ""
    elif colon_count == 1:
        module, obj_name = name.split(":")
    else:
        raise ValueError(f"Too many colons in '{name}'")

    module_obj = import_module_or_file(module)
    if not obj_name:
        if not module_obj.__all__:
            raise ValueError(f"Failed to find default object in {module}")
        obj_name = module_obj.__all__[0]
    obj = getattr(module_obj, obj_name)
    if obj is None:
        raise ValueError("Estimator not found")
    return obj


def make_network(system: System, network: Mapping) -> nn.Module:
    network_opts = {**network}  # Shallow copy since we are going to popping stuffs
    network_class = resolve_object(network_opts.pop("type"))
    return network_class(system=system, **network_opts)
