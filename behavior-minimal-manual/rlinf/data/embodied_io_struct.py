# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional

import torch

from rlinf.utils.nested_dict_process import put_tensor_device


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single eval chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None
    terminations: Optional[torch.Tensor] = None
    truncations: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    env_infos: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )

    @staticmethod
    def prepare_observations(obs: dict[str, Any]) -> dict[str, Any]:
        task_descriptions = obs.get("task_descriptions", None)
        return {
            "main_images": obs.get("main_images", None),
            "wrist_images": obs.get("wrist_images", None),
            "extra_view_images": obs.get("extra_view_images", None),
            "states": obs.get("states", None),
            "task_descriptions": list(task_descriptions)
            if task_descriptions is not None
            else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs": self.prepare_observations(self.obs),
            "final_obs": self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None,
            "dones": self.dones,
            "terminations": self.terminations,
            "truncations": self.truncations,
            "rewards": self.rewards,
            "env_infos": self.env_infos,
        }
