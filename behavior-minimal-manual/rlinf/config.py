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

import dataclasses
import logging
import os
from dataclasses import asdict
from typing import ClassVar, Optional, Union

import torch
import yaml
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.envs import SupportedEnvType
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import HybridComponentPlacement

logging.getLogger().setLevel(logging.INFO)


@dataclasses.dataclass(frozen=True)
class SupportedModel:
    value: str

    models: ClassVar[dict[str, "SupportedModel"]] = {}

    @classmethod
    def register(cls, value: str, force: bool = False) -> "SupportedModel":
        if not value:
            raise ValueError("model_type must be a non-empty string.")
        if value in cls.models:
            if not force:
                raise ValueError(f"Model type `{value}` is already registered.")
        else:
            cls.models[value] = cls.__private_create__(value)
        return cls.models[value]

    @classmethod
    def get(cls, value: str) -> "SupportedModel":
        if value not in cls.models:
            raise NotImplementedError(
                f"Model Type: {value} not supported. Supported models: {sorted(cls.models)}"
            )
        return cls.models[value]

    def __new__(cls, value: str):
        return cls.get(value)

    @classmethod
    def __private_create__(cls, value: str) -> "SupportedModel":
        obj = object.__new__(cls)
        object.__setattr__(obj, "value", value)
        return obj


SupportedModel.OPENPI = SupportedModel.register("openpi", force=True)

EMBODIED_MODEL = {SupportedModel.OPENPI}
SUPPORTED_TASK_TYPE = ["embodied"]
SUPPORTED_TRAINING_BACKENDS = ["fsdp"]

__all__ = ["build_config", "validate_cfg", "SupportedModel"]


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    if precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    if precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    if precision in [None, "null"]:
        return None
    raise ValueError(f"Could not parse precision `{precision}` to a torch.dtype")


def validate_fsdp_cfg(cfg: DictConfig) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        param_dtype = config.mixed_precision.param_dtype
        reduce_dtype = config.mixed_precision.reduce_dtype
        buffer_dtype = config.mixed_precision.buffer_dtype

        all_none = param_dtype is None and reduce_dtype is None and buffer_dtype is None
        all_fp32 = (
            param_dtype == "fp32" and reduce_dtype == "fp32" and buffer_dtype == "fp32"
        )
        use_fsdp_mixed_precision = not (all_none or all_fp32)

        amp_autocast = config.get("amp_autocast", {})
        config.amp_autocast = {
            "enabled": amp_autocast.get("enabled", False),
            "precision": amp_autocast.get("precision", "bf16"),
        }
        grad_scaler = config.get("grad_scaler", {})
        config.grad_scaler = {
            "enabled": grad_scaler.get("enabled", False),
            "init_scale": grad_scaler.get("init_scale", None),
            "growth_interval": grad_scaler.get("growth_interval", None),
        }

        if config.amp_autocast.enabled and use_fsdp_mixed_precision:
            raise AssertionError(
                "amp_autocast should not be enabled when fsdp mixed_precision is enabled"
            )
        assert config.amp_autocast.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp_autocast.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")
        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )
        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )
        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )
        assert cfg.fsdp_config.backward_prefetch in [None, "pre", "post"], (
            "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"
        )
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )
        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", None
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", None
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", None
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

    return cfg


def _validate_env_counts(cfg: DictConfig, split: str) -> None:
    component_placement = HybridComponentPlacement(cfg, Cluster())
    env_world_size = component_placement.get_world_size("env")
    stage_num = cfg.rollout.pipeline_stage_num
    env_cfg = cfg.env[split]

    assert env_cfg.total_num_envs > 0, (
        f"Total number of parallel environments for {split} must be greater than 0"
    )
    assert env_cfg.total_num_envs % env_world_size == 0, (
        f"Total number of parallel environments for {split} must be divisible by the number of environment processes"
    )
    assert env_cfg.total_num_envs % env_world_size % stage_num == 0, (
        f"Total number of parallel environments for {split} must be divisible by env processes and pipeline stages"
    )
    assert env_cfg.total_num_envs // env_world_size // stage_num > 0, (
        f"env.{split}.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
    )
    assert env_cfg.total_num_envs // env_world_size // stage_num % env_cfg.group_size == 0, (
        f"env.{split}.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by group_size"
    )
    assert env_cfg.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks == 0, (
        f"env.{split}.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
    )


def _attach_behavior_config(cfg: DictConfig) -> None:
    import omnigibson as og

    assert cfg.env.train.base_config_name == "r1pro_behavior", (
        f"Only r1pro_behavior is supported, got {cfg.env.train.base_config_name}"
    )
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    with open(config_filename, "r") as f:
        omnigibson_cfg = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    with open_dict(omnigibson_cfg):
        omnigibson_cfg.robots[0].obs_modalities = ["rgb", "depth", "proprio"]
    cfg.env.train.omnigibson_cfg = omnigibson_cfg
    cfg.env.eval.omnigibson_cfg = omnigibson_cfg


def validate_embodied_cfg(cfg: DictConfig) -> DictConfig:
    model_type = SupportedModel(cfg.actor.model.model_type)
    assert model_type in EMBODIED_MODEL, (
        f"Model type: '{cfg.actor.model.model_type}' is not an embodied model. "
        f"Supported embodied models: {sorted([x.value for x in EMBODIED_MODEL])}."
    )

    if not cfg.runner.get("only_eval", False) and cfg.algorithm.loss_type in (
        "actor_critic",
        "decoupled_actor_critic",
    ):
        assert cfg.actor.model.get("add_value_head", False), (
            "actor.model.add_value_head must be True when training actor_critic PPO"
        )

    if cfg.runner.val_check_interval > 0 or cfg.runner.get("only_eval", False):
        _validate_env_counts(cfg, "eval")
    if not cfg.runner.get("only_eval", False):
        _validate_env_counts(cfg, "train")

    with open_dict(cfg):
        weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        assert weight_sync_interval > 0, "weight_sync_interval must be greater than 0"
        cfg.runner.weight_sync_interval = weight_sync_interval
        cfg.runner.overlap_env_bootstrap = bool(
            cfg.runner.get("overlap_env_bootstrap", False)
        ) and not cfg.env.train.get("enable_offload", False)

        assert SupportedEnvType(cfg.env.train.env_type) == SupportedEnvType.BEHAVIOR
        assert SupportedEnvType(cfg.env.eval.env_type) == SupportedEnvType.BEHAVIOR
        _attach_behavior_config(cfg)

    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.runner.per_worker_log = cfg.runner.get("per_worker_log", False)
        cfg.runner.per_worker_log_path = None
        if cfg.runner.per_worker_log:
            cfg.runner.per_worker_log_path = os.path.join(
                cfg.runner.logger.log_path, "worker_logs"
            )
        profiling_cfg = cfg.cluster.get("profiling", None)
        if profiling_cfg is not None and bool(profiling_cfg.get("enabled", True)):
            if not profiling_cfg.get("output_dir", None):
                cfg.cluster.profiling.output_dir = os.path.abspath(
                    os.path.join(
                        cfg.runner.logger.log_path,
                        cfg.runner.logger.experiment_name,
                        "profiling",
                    )
                )

    Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    cfg = validate_embodied_cfg(cfg)

    if cfg.actor.training_backend not in SUPPORTED_TRAINING_BACKENDS:
        raise AssertionError(
            f"Unsupported training_backend {cfg.actor.training_backend}. "
            f"Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
        )

    component_placement = HybridComponentPlacement(cfg, Cluster())
    actor_world_size = component_placement.get_world_size("actor")
    assert cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * actor_world_size) == 0, (
        f"actor.global_batch_size ({cfg.actor.global_batch_size}) must be divisible by "
        f"actor.micro_batch_size ({cfg.actor.micro_batch_size}) * actor_world_size ({actor_world_size})"
    )
    cfg.actor = validate_fsdp_cfg(cfg.actor)

    if cfg.get("critic", None) is not None and cfg.critic.use_critic_model:
        assert cfg.critic.training_backend == "fsdp", "Only FSDP critic is supported"
        cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)
