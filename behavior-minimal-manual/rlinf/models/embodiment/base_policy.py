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

from abc import ABC, abstractmethod
from enum import Enum


class ForwardType(Enum):
    DEFAULT = "default"


class BasePolicy(ABC):
    """
    Base interface for all policies.

    Subclasses must implement:
        - forward
        - default_forward
        - predict_action_batch

    """

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def default_forward(self, **kwargs): ...

    @abstractmethod
    def predict_action_batch(self, **kwargs): ...

    def enable_torch_compile(
        self,
        mode: str = "max-autotune-no-cudagraphs",
    ):
        raise NotImplementedError(
            "torch compile is not supported for current policy, please set `enable_torch_compile=False` for now"
        )
