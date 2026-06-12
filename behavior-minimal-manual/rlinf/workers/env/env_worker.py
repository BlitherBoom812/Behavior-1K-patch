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

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.nested_dict_process import split_dict, update_nested_cfg
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg
        self.eval_env_list = []
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.eval_num_envs_per_stage = (
            self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.eval_prev_done: list[torch.Tensor] = [
            torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
            for _ in range(self.stage_num)
        ]

    def init_worker(self):
        self.dst_rank_map = self._setup_dst_rank_map()
        self.src_rank_map = self._setup_src_rank_map()
        self.log_info(f"Env worker initialized with dst_rank_map: {self.dst_rank_map}")
        self.log_info(f"Env worker initialized with src_rank_map: {self.src_rank_map}")

        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )
        self.update_env_cfg()

        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
        self.eval_env_list = self._setup_env_and_wrappers(
            env_cls=eval_env_cls,
            env_cfg=self.cfg.env.eval,
            num_envs_per_stage=self.eval_num_envs_per_stage,
        )

    def update_env_cfg(self):
        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is None:
            return
        assert len(eval_override_cfgs) > self._rank, (
            f"{len(eval_override_cfgs)=} > {self._rank=}"
        )
        general_eval_override_cfg = OmegaConf.to_container(
            self.cfg.env.eval.get("override_cfg", {}), resolve=True
        )
        eval_override_cfg = OmegaConf.to_container(
            eval_override_cfgs[self._rank], resolve=True
        ).copy()
        base_eval_cfg = {}
        base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
        base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
        setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []
        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            env_list.append(env)
        return env_list

    def _setup_dst_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        return {
            "rollout_eval": CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                src_world_size=self._component_placement.get_world_size("env"),
                dst_world_size=self._component_placement.get_world_size("rollout"),
                src_rank=self._rank,
            )
        }

    def _setup_src_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        return {
            "rollout_eval": CommMapper.get_src_ranks(
                batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                src_world_size=self._component_placement.get_world_size("rollout"),
                dst_world_size=self._component_placement.get_world_size("env"),
                dst_rank=self._rank,
            )
        }

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}
        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            infos["final_observation"]
            if isinstance(infos, dict) and "final_observation" in infos
            else None
        )

        current_dones = chunk_dones[:, -1]
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any() and isinstance(infos, dict):
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        return EnvOutput(obs=extracted_obs, final_obs=final_obs), env_info

    @Worker.timer("env/recv_actions")
    def recv_chunk_actions(self, input_channel: Channel, mode="eval") -> np.ndarray:
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        return np.concatenate(chunk_action, axis=0)

    @Worker.timer("env/send_obs")
    def send_env_batch(
        self,
        rollout_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["eval"] = "eval",
    ) -> None:
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = split_dict(env_batch, split_sizes)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            rollout_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def finish_rollout(self):
        for i in range(self.stage_num):
            if self.cfg.env.eval.video_cfg.save_video and isinstance(
                self.eval_env_list[i], RecordVideo
            ):
                self.eval_env_list[i].flush_video()
            if not self.cfg.env.eval.auto_reset:
                self.eval_env_list[i].update_reset_state_ids()

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]},
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )
                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    is_last_step = eval_step == self.n_eval_chunk_steps - 1
                    is_last_epoch = (
                        eval_rollout_epoch == self.cfg.algorithm.eval_rollout_epoch - 1
                    )
                    if is_last_step and (self.cfg.env.eval.auto_reset and is_last_epoch):
                        continue
                    if is_last_step and not self.cfg.env.eval.auto_reset:
                        continue

                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]},
                    )

            self.finish_rollout()

        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()
        return eval_metrics
