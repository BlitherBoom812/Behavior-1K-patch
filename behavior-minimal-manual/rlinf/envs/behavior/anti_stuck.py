# Copyright 2025 The RLinf Authors.
#
# Local BEHAVIOR eval helper used only when ANTI_STUCK_ENABLE=1.

from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch


class AntiStuckWrapper:
    """Websocket-eval compatible anti-stuck state machine for BEHAVIOR.

    The wrapper observes robot base XY and policy actions. When the robot stays
    almost fixed while policy actions continue to request base translation, or
    when the whole robot is idle for a long window, it temporarily overrides the
    base action dimensions with a lateral-then-forward escape macro. All
    non-base action dimensions are preserved from the policy output.
    """

    def __init__(
        self,
        task_name: str,
        version: str = "unversioned",
        enabled: bool = False,
        tasks: str = "all",
        position_window: int = 60,
        position_threshold: float = 0.08,
        base_command_threshold: float = 0.03,
        intent_ratio_threshold: float = 0.45,
        repeat_window: int = 24,
        repeat_std_threshold: float = 0.008,
        confirm_steps: int = 10,
        strong_base_command_threshold: float = 0.12,
        strong_position_scale: float = 0.5,
        low_progress_command_threshold: float = 0.24,
        low_progress_position_threshold: float = 0.30,
        low_progress_position_scale: float = 0.50,
        idle_position_window: int = 0,
        idle_position_threshold: float = 0.05,
        idle_intent_ratio_threshold: float = 0.05,
        idle_confirm_steps: int = 20,
        idle_nonbase_delta_threshold: float = 0.03,
        idle_nonbase_delta_ratio_threshold: float = 0.02,
        manipulation_guard_window: int = 60,
        manipulation_eef_delta_threshold: float = 0.01,
        manipulation_eef_delta_ratio_threshold: float = 0.15,
        manipulation_eef_only_ratio_threshold: float = 0.90,
        manipulation_gripper_delta_threshold: float = 0.005,
        manipulation_gripper_delta_ratio_threshold: float = 0.10,
        arm_deployed_guard_window: int = 0,
        arm_deployed_abs_threshold: float = 0.40,
        arm_deployed_ratio_threshold: float = 0.60,
        arm_deployed_base_command_threshold: float = 0.35,
        initial_cooldown_steps: int = 0,
        cooldown_steps: int = 2000,
        ineffective_cooldown_steps: int = 300,
        escape_steps: int = 40,
        escape_lateral_fraction: float = 0.5,
        max_triggers: int = 1,
        back_vel: float = -0.16,
        strafe_vel: float = 0.18,
        yaw_vel: float = 0.30,
        forward_vel: float = 0.14,
        escape_scale_growth: float = 0.0,
        escape_min_displacement: float = 0.10,
        escape_max_displacement: float = 0.0,
        escape_mode: str = "legacy",
        lateral_probe_distance: float = 0.30,
        lateral_probe_steps: int = 60,
        forward_probe_distance: float = 0.12,
        forward_probe_steps: int = 45,
        forward_commit_distance: float = 0.80,
        forward_commit_steps: int = 140,
        forward_commit_stall_check_steps: int = 30,
        forward_commit_min_progress: float = 0.08,
        max_probe_cycles: int = 12,
        forward_try_distance: float = 0.45,
        forward_try_steps: int = 90,
        forward_stall_check_steps: int = 30,
        forward_stall_min_progress: float = 0.08,
        strafe_step_distance: float = 0.30,
        strafe_step_steps: int = 60,
        max_strafe_cycles: int = 12,
        direction_return_steps: int = 60,
        direction_return_tolerance: float = 0.08,
        direction_min_probe_progress: float = 0.08,
        direction_commit_lateral_distance: float = 0.60,
        direction_commit_lateral_steps: int = 120,
        direction_commit_forward_distance: float = 1.00,
        direction_commit_forward_steps: int = 180,
        direction_stall_check_steps: int = 45,
        direction_stall_min_progress: float = 0.12,
        direction_max_side_switches: int = 1,
        retreat_distance: float = 0.70,
        retreat_steps: int = 140,
        retreat_stall_check_steps: int = 45,
        retreat_stall_min_progress: float = 0.10,
        crouch_enable: bool = False,
        crouch_target: str | np.ndarray = "1.25,-1.90,-0.65,0.0",
        crouch_steps: int = 30,
        tilt_recovery_enable: bool = False,
        tilt_warn_rad: float = 0.35,
        tilt_clear_rad: float = 0.20,
        tilt_recovery_steps: int = 45,
        tilt_recovery_min_steps: int = 12,
        tilt_recovery_scale: float = 0.70,
        tilt_recovery_max_attempts: int = 3,
        task_progress_guard_enable: bool = False,
        task_progress_object_names: str = "",
        task_progress_distance: float = 1.20,
        task_progress_sticky_steps: int = 300,
        log_prefix: str = "",
    ) -> None:
        allowed_tasks = {x.strip() for x in tasks.split(",") if x.strip()}
        self.version = str(version)
        self.enabled = enabled and ("all" in allowed_tasks or task_name in allowed_tasks)
        self.position_window = int(position_window)
        self.position_threshold = float(position_threshold)
        self.base_command_threshold = float(base_command_threshold)
        self.intent_ratio_threshold = float(intent_ratio_threshold)
        self.repeat_window = int(repeat_window)
        self.repeat_std_threshold = float(repeat_std_threshold)
        self.confirm_steps = int(confirm_steps)
        self.strong_base_command_threshold = float(strong_base_command_threshold)
        self.strong_position_scale = float(strong_position_scale)
        self.low_progress_command_threshold = float(low_progress_command_threshold)
        self.low_progress_position_threshold = float(low_progress_position_threshold)
        self.low_progress_position_scale = float(low_progress_position_scale)
        self.idle_position_window = int(idle_position_window)
        self.idle_position_threshold = float(idle_position_threshold)
        self.idle_intent_ratio_threshold = float(idle_intent_ratio_threshold)
        self.idle_confirm_steps = int(idle_confirm_steps)
        self.idle_nonbase_delta_threshold = float(idle_nonbase_delta_threshold)
        self.idle_nonbase_delta_ratio_threshold = float(idle_nonbase_delta_ratio_threshold)
        self.manipulation_guard_window = int(manipulation_guard_window)
        self.manipulation_eef_delta_threshold = float(manipulation_eef_delta_threshold)
        self.manipulation_eef_delta_ratio_threshold = float(manipulation_eef_delta_ratio_threshold)
        self.manipulation_eef_only_ratio_threshold = float(manipulation_eef_only_ratio_threshold)
        self.manipulation_gripper_delta_threshold = float(manipulation_gripper_delta_threshold)
        self.manipulation_gripper_delta_ratio_threshold = float(manipulation_gripper_delta_ratio_threshold)
        self.arm_deployed_guard_window = int(arm_deployed_guard_window)
        self.arm_deployed_abs_threshold = float(arm_deployed_abs_threshold)
        self.arm_deployed_ratio_threshold = float(arm_deployed_ratio_threshold)
        self.arm_deployed_base_command_threshold = float(arm_deployed_base_command_threshold)
        self.initial_cooldown_steps = int(initial_cooldown_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.ineffective_cooldown_steps = int(ineffective_cooldown_steps)
        self.escape_steps = int(escape_steps)
        self.escape_lateral_fraction = min(0.9, max(0.1, float(escape_lateral_fraction)))
        self.max_triggers = int(max_triggers)
        self.back_vel = float(back_vel)
        self.strafe_vel = float(strafe_vel)
        self.yaw_vel = float(yaw_vel)
        self.forward_vel = float(forward_vel)
        self.escape_scale_growth = float(escape_scale_growth)
        self.escape_min_displacement = float(escape_min_displacement)
        self.escape_max_displacement = float(escape_max_displacement)
        self.escape_mode = str(escape_mode)
        self.lateral_probe_distance = float(lateral_probe_distance)
        self.lateral_probe_steps = int(lateral_probe_steps)
        self.forward_probe_distance = float(forward_probe_distance)
        self.forward_probe_steps = int(forward_probe_steps)
        self.forward_commit_distance = float(forward_commit_distance)
        self.forward_commit_steps = int(forward_commit_steps)
        self.forward_commit_stall_check_steps = int(forward_commit_stall_check_steps)
        self.forward_commit_min_progress = float(forward_commit_min_progress)
        self.max_probe_cycles = int(max_probe_cycles)
        self.forward_try_distance = float(forward_try_distance)
        self.forward_try_steps = int(forward_try_steps)
        self.forward_stall_check_steps = int(forward_stall_check_steps)
        self.forward_stall_min_progress = float(forward_stall_min_progress)
        self.strafe_step_distance = float(strafe_step_distance)
        self.strafe_step_steps = int(strafe_step_steps)
        self.max_strafe_cycles = int(max_strafe_cycles)
        self.direction_return_steps = int(direction_return_steps)
        self.direction_return_tolerance = float(direction_return_tolerance)
        self.direction_min_probe_progress = float(direction_min_probe_progress)
        self.direction_commit_lateral_distance = float(direction_commit_lateral_distance)
        self.direction_commit_lateral_steps = int(direction_commit_lateral_steps)
        self.direction_commit_forward_distance = float(direction_commit_forward_distance)
        self.direction_commit_forward_steps = int(direction_commit_forward_steps)
        self.direction_stall_check_steps = int(direction_stall_check_steps)
        self.direction_stall_min_progress = float(direction_stall_min_progress)
        self.direction_max_side_switches = int(direction_max_side_switches)
        self.retreat_distance = float(retreat_distance)
        self.retreat_steps = int(retreat_steps)
        self.retreat_stall_check_steps = int(retreat_stall_check_steps)
        self.retreat_stall_min_progress = float(retreat_stall_min_progress)
        self.crouch_enable = bool(crouch_enable)
        self.crouch_target = self._parse_vector(crouch_target, np.array([1.25, -1.90, -0.65, 0.0], dtype=np.float64))
        self.crouch_steps = int(crouch_steps)
        self.tilt_recovery_enable = bool(tilt_recovery_enable)
        self.tilt_warn_rad = float(tilt_warn_rad)
        self.tilt_clear_rad = float(tilt_clear_rad)
        self.tilt_recovery_steps = int(tilt_recovery_steps)
        self.tilt_recovery_min_steps = int(tilt_recovery_min_steps)
        self.tilt_recovery_scale = float(tilt_recovery_scale)
        self.tilt_recovery_max_attempts = int(tilt_recovery_max_attempts)
        self.task_progress_guard_enable = bool(task_progress_guard_enable)
        self.task_progress_object_names = tuple(
            x.strip() for x in str(task_progress_object_names).split(",") if x.strip()
        )
        self.task_progress_distance = float(task_progress_distance)
        self.task_progress_sticky_steps = int(task_progress_sticky_steps)
        self.log_prefix = str(log_prefix)

        history_len = max(
            self.position_window,
            self.repeat_window,
            self.idle_position_window,
            self.arm_deployed_guard_window,
            1,
        )
        self.position_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)
        self.base_action_history = deque(maxlen=history_len)
        self.nonbase_action_history = deque(maxlen=history_len)
        self.eef_position_history = deque(maxlen=history_len)
        self.gripper_position_history = deque(maxlen=history_len)
        self.reset()
        self._log(
            "version=%s, enabled=%s, position_window=%d, position_threshold=%.3f, "
            "base_command_threshold=%.3f, intent_ratio_threshold=%.2f, escape_steps=%d, "
            "lateral_fraction=%.2f, max_triggers=%d, arm_deployed_guard_window=%d, "
            "escape_max_displacement=%.3f, escape_mode=%s, lateral_probe_distance=%.3f, "
            "forward_probe_distance=%.3f, forward_commit_distance=%.3f, "
            "forward_commit_stall_check_steps=%d, forward_commit_min_progress=%.3f, "
            "max_probe_cycles=%d, forward_try_distance=%.3f, forward_try_steps=%d, "
            "forward_stall_check_steps=%d, forward_stall_min_progress=%.3f, "
            "strafe_step_distance=%.3f, strafe_step_steps=%d, max_strafe_cycles=%d, "
            "direction_return_steps=%d, direction_return_tolerance=%.3f, "
            "direction_min_probe_progress=%.3f, direction_commit_lateral_distance=%.3f, "
            "direction_commit_lateral_steps=%d, direction_commit_forward_distance=%.3f, "
            "direction_commit_forward_steps=%d, direction_stall_check_steps=%d, "
            "direction_stall_min_progress=%.3f, direction_max_side_switches=%d, "
            "retreat_distance=%.3f, retreat_steps=%d, retreat_stall_check_steps=%d, "
            "retreat_stall_min_progress=%.3f, "
            "crouch_enable=%s, crouch_target=%s, crouch_steps=%d, "
            "tilt_recovery_enable=%s, tilt_warn_rad=%.3f, tilt_clear_rad=%.3f, "
            "tilt_recovery_steps=%d, tilt_recovery_scale=%.2f, tilt_recovery_max_attempts=%d, "
            "task_progress_guard_enable=%s, task_progress_objects=%s, "
            "task_progress_distance=%.3f, task_progress_sticky_steps=%d",
            self.version,
            self.enabled,
            self.position_window,
            self.position_threshold,
            self.base_command_threshold,
            self.intent_ratio_threshold,
            self.escape_steps,
            self.escape_lateral_fraction,
            self.max_triggers,
            self.arm_deployed_guard_window,
            self.escape_max_displacement,
            self.escape_mode,
            self.lateral_probe_distance,
            self.forward_probe_distance,
            self.forward_commit_distance,
            self.forward_commit_stall_check_steps,
            self.forward_commit_min_progress,
            self.max_probe_cycles,
            self.forward_try_distance,
            self.forward_try_steps,
            self.forward_stall_check_steps,
            self.forward_stall_min_progress,
            self.strafe_step_distance,
            self.strafe_step_steps,
            self.max_strafe_cycles,
            self.direction_return_steps,
            self.direction_return_tolerance,
            self.direction_min_probe_progress,
            self.direction_commit_lateral_distance,
            self.direction_commit_lateral_steps,
            self.direction_commit_forward_distance,
            self.direction_commit_forward_steps,
            self.direction_stall_check_steps,
            self.direction_stall_min_progress,
            self.direction_max_side_switches,
            self.retreat_distance,
            self.retreat_steps,
            self.retreat_stall_check_steps,
            self.retreat_stall_min_progress,
            self.crouch_enable,
            np.round(self.crouch_target, 4).tolist(),
            self.crouch_steps,
            self.tilt_recovery_enable,
            self.tilt_warn_rad,
            self.tilt_clear_rad,
            self.tilt_recovery_steps,
            self.tilt_recovery_scale,
            self.tilt_recovery_max_attempts,
            self.task_progress_guard_enable,
            list(self.task_progress_object_names),
            self.task_progress_distance,
            self.task_progress_sticky_steps,
        )

    @classmethod
    def from_env(cls, task_name: str, log_prefix: str = "") -> "AntiStuckWrapper":
        enabled = os.environ.get("ANTI_STUCK_ENABLE", "0").lower() in {"1", "true", "yes"}
        return cls(
            task_name=task_name,
            version=os.environ.get("ANTI_STUCK_VERSION", "unversioned"),
            enabled=enabled,
            tasks=os.environ.get("ANTI_STUCK_TASKS", "all"),
            position_window=int(os.environ.get("ANTI_STUCK_POSITION_WINDOW", "60")),
            position_threshold=float(os.environ.get("ANTI_STUCK_POSITION_THRESHOLD", "0.08")),
            base_command_threshold=float(os.environ.get("ANTI_STUCK_BASE_COMMAND_THRESHOLD", "0.03")),
            intent_ratio_threshold=float(os.environ.get("ANTI_STUCK_INTENT_RATIO_THRESHOLD", "0.45")),
            repeat_window=int(os.environ.get("ANTI_STUCK_REPEAT_WINDOW", "24")),
            repeat_std_threshold=float(os.environ.get("ANTI_STUCK_REPEAT_STD_THRESHOLD", "0.008")),
            confirm_steps=int(os.environ.get("ANTI_STUCK_CONFIRM_STEPS", "10")),
            strong_base_command_threshold=float(os.environ.get("ANTI_STUCK_STRONG_BASE_COMMAND_THRESHOLD", "0.12")),
            strong_position_scale=float(os.environ.get("ANTI_STUCK_STRONG_POSITION_SCALE", "0.5")),
            low_progress_command_threshold=float(os.environ.get("ANTI_STUCK_LOW_PROGRESS_COMMAND_THRESHOLD", "0.24")),
            low_progress_position_threshold=float(os.environ.get("ANTI_STUCK_LOW_PROGRESS_POSITION_THRESHOLD", "0.30")),
            low_progress_position_scale=float(os.environ.get("ANTI_STUCK_LOW_PROGRESS_POSITION_SCALE", "0.50")),
            idle_position_window=int(os.environ.get("ANTI_STUCK_IDLE_POSITION_WINDOW", "0")),
            idle_position_threshold=float(os.environ.get("ANTI_STUCK_IDLE_POSITION_THRESHOLD", "0.05")),
            idle_intent_ratio_threshold=float(os.environ.get("ANTI_STUCK_IDLE_INTENT_RATIO_THRESHOLD", "0.05")),
            idle_confirm_steps=int(os.environ.get("ANTI_STUCK_IDLE_CONFIRM_STEPS", "20")),
            idle_nonbase_delta_threshold=float(os.environ.get("ANTI_STUCK_IDLE_NONBASE_DELTA_THRESHOLD", "0.03")),
            idle_nonbase_delta_ratio_threshold=float(os.environ.get("ANTI_STUCK_IDLE_NONBASE_DELTA_RATIO_THRESHOLD", "0.02")),
            manipulation_guard_window=int(os.environ.get("ANTI_STUCK_MANIPULATION_GUARD_WINDOW", "60")),
            manipulation_eef_delta_threshold=float(os.environ.get("ANTI_STUCK_MANIPULATION_EEF_DELTA_THRESHOLD", "0.01")),
            manipulation_eef_delta_ratio_threshold=float(os.environ.get("ANTI_STUCK_MANIPULATION_EEF_DELTA_RATIO_THRESHOLD", "0.15")),
            manipulation_eef_only_ratio_threshold=float(os.environ.get("ANTI_STUCK_MANIPULATION_EEF_ONLY_RATIO_THRESHOLD", "0.90")),
            manipulation_gripper_delta_threshold=float(os.environ.get("ANTI_STUCK_MANIPULATION_GRIPPER_DELTA_THRESHOLD", "0.005")),
            manipulation_gripper_delta_ratio_threshold=float(os.environ.get("ANTI_STUCK_MANIPULATION_GRIPPER_DELTA_RATIO_THRESHOLD", "0.10")),
            arm_deployed_guard_window=int(os.environ.get("ANTI_STUCK_ARM_DEPLOYED_GUARD_WINDOW", "0")),
            arm_deployed_abs_threshold=float(os.environ.get("ANTI_STUCK_ARM_DEPLOYED_ABS_THRESHOLD", "0.40")),
            arm_deployed_ratio_threshold=float(os.environ.get("ANTI_STUCK_ARM_DEPLOYED_RATIO_THRESHOLD", "0.60")),
            arm_deployed_base_command_threshold=float(os.environ.get("ANTI_STUCK_ARM_DEPLOYED_BASE_COMMAND_THRESHOLD", "0.35")),
            initial_cooldown_steps=int(os.environ.get("ANTI_STUCK_INITIAL_COOLDOWN_STEPS", "0")),
            cooldown_steps=int(os.environ.get("ANTI_STUCK_COOLDOWN_STEPS", "2000")),
            ineffective_cooldown_steps=int(os.environ.get("ANTI_STUCK_INEFFECTIVE_COOLDOWN_STEPS", "300")),
            escape_steps=int(os.environ.get("ANTI_STUCK_ESCAPE_STEPS", "40")),
            escape_lateral_fraction=float(os.environ.get("ANTI_STUCK_LATERAL_FRACTION", "0.5")),
            max_triggers=int(os.environ.get("ANTI_STUCK_MAX_TRIGGERS", "1")),
            back_vel=float(os.environ.get("ANTI_STUCK_BACK", "-0.16")),
            strafe_vel=float(os.environ.get("ANTI_STUCK_STRAFE", "0.18")),
            yaw_vel=float(os.environ.get("ANTI_STUCK_YAW", "0.30")),
            forward_vel=float(os.environ.get("ANTI_STUCK_FORWARD", "0.14")),
            escape_scale_growth=float(os.environ.get("ANTI_STUCK_ESCAPE_SCALE_GROWTH", "0.0")),
            escape_min_displacement=float(os.environ.get("ANTI_STUCK_ESCAPE_MIN_DISPLACEMENT", "0.10")),
            escape_max_displacement=float(os.environ.get("ANTI_STUCK_ESCAPE_MAX_DISPLACEMENT", "0.0")),
            escape_mode=os.environ.get("ANTI_STUCK_ESCAPE_MODE", "legacy"),
            lateral_probe_distance=float(os.environ.get("ANTI_STUCK_LATERAL_PROBE_DISTANCE", "0.30")),
            lateral_probe_steps=int(os.environ.get("ANTI_STUCK_LATERAL_PROBE_STEPS", "60")),
            forward_probe_distance=float(os.environ.get("ANTI_STUCK_FORWARD_PROBE_DISTANCE", "0.12")),
            forward_probe_steps=int(os.environ.get("ANTI_STUCK_FORWARD_PROBE_STEPS", "45")),
            forward_commit_distance=float(os.environ.get("ANTI_STUCK_FORWARD_COMMIT_DISTANCE", "0.80")),
            forward_commit_steps=int(os.environ.get("ANTI_STUCK_FORWARD_COMMIT_STEPS", "140")),
            forward_commit_stall_check_steps=int(os.environ.get("ANTI_STUCK_FORWARD_COMMIT_STALL_CHECK_STEPS", "30")),
            forward_commit_min_progress=float(os.environ.get("ANTI_STUCK_FORWARD_COMMIT_MIN_PROGRESS", "0.08")),
            max_probe_cycles=int(os.environ.get("ANTI_STUCK_MAX_PROBE_CYCLES", "12")),
            forward_try_distance=float(os.environ.get("ANTI_STUCK_FORWARD_TRY_DISTANCE", "0.45")),
            forward_try_steps=int(os.environ.get("ANTI_STUCK_FORWARD_TRY_STEPS", "90")),
            forward_stall_check_steps=int(os.environ.get("ANTI_STUCK_FORWARD_STALL_CHECK_STEPS", "30")),
            forward_stall_min_progress=float(os.environ.get("ANTI_STUCK_FORWARD_STALL_MIN_PROGRESS", "0.08")),
            strafe_step_distance=float(os.environ.get("ANTI_STUCK_STRAFE_STEP_DISTANCE", "0.30")),
            strafe_step_steps=int(os.environ.get("ANTI_STUCK_STRAFE_STEP_STEPS", "60")),
            max_strafe_cycles=int(os.environ.get("ANTI_STUCK_MAX_STRAFE_CYCLES", "12")),
            direction_return_steps=int(os.environ.get("ANTI_STUCK_DIRECTION_RETURN_STEPS", "60")),
            direction_return_tolerance=float(os.environ.get("ANTI_STUCK_DIRECTION_RETURN_TOLERANCE", "0.08")),
            direction_min_probe_progress=float(os.environ.get("ANTI_STUCK_DIRECTION_MIN_PROBE_PROGRESS", "0.08")),
            direction_commit_lateral_distance=float(
                os.environ.get("ANTI_STUCK_DIRECTION_COMMIT_LATERAL_DISTANCE", "0.60")
            ),
            direction_commit_lateral_steps=int(os.environ.get("ANTI_STUCK_DIRECTION_COMMIT_LATERAL_STEPS", "120")),
            direction_commit_forward_distance=float(
                os.environ.get("ANTI_STUCK_DIRECTION_COMMIT_FORWARD_DISTANCE", "1.00")
            ),
            direction_commit_forward_steps=int(os.environ.get("ANTI_STUCK_DIRECTION_COMMIT_FORWARD_STEPS", "180")),
            direction_stall_check_steps=int(os.environ.get("ANTI_STUCK_DIRECTION_STALL_CHECK_STEPS", "45")),
            direction_stall_min_progress=float(os.environ.get("ANTI_STUCK_DIRECTION_STALL_MIN_PROGRESS", "0.12")),
            direction_max_side_switches=int(os.environ.get("ANTI_STUCK_DIRECTION_MAX_SIDE_SWITCHES", "1")),
            retreat_distance=float(os.environ.get("ANTI_STUCK_RETREAT_DISTANCE", "0.70")),
            retreat_steps=int(os.environ.get("ANTI_STUCK_RETREAT_STEPS", "140")),
            retreat_stall_check_steps=int(os.environ.get("ANTI_STUCK_RETREAT_STALL_CHECK_STEPS", "45")),
            retreat_stall_min_progress=float(os.environ.get("ANTI_STUCK_RETREAT_STALL_MIN_PROGRESS", "0.10")),
            crouch_enable=os.environ.get("ANTI_STUCK_CROUCH_ENABLE", "0").lower() in {"1", "true", "yes"},
            crouch_target=os.environ.get("ANTI_STUCK_CROUCH_TARGET", "1.25,-1.90,-0.65,0.0"),
            crouch_steps=int(os.environ.get("ANTI_STUCK_CROUCH_STEPS", "30")),
            tilt_recovery_enable=os.environ.get("ANTI_STUCK_TILT_RECOVERY_ENABLE", "0").lower()
            in {"1", "true", "yes"},
            tilt_warn_rad=float(os.environ.get("ANTI_STUCK_TILT_WARN_RAD", "0.35")),
            tilt_clear_rad=float(os.environ.get("ANTI_STUCK_TILT_CLEAR_RAD", "0.20")),
            tilt_recovery_steps=int(os.environ.get("ANTI_STUCK_TILT_RECOVERY_STEPS", "45")),
            tilt_recovery_min_steps=int(os.environ.get("ANTI_STUCK_TILT_RECOVERY_MIN_STEPS", "12")),
            tilt_recovery_scale=float(os.environ.get("ANTI_STUCK_TILT_RECOVERY_SCALE", "0.70")),
            tilt_recovery_max_attempts=int(os.environ.get("ANTI_STUCK_TILT_RECOVERY_MAX_ATTEMPTS", "3")),
            task_progress_guard_enable=os.environ.get("ANTI_STUCK_TASK_PROGRESS_GUARD_ENABLE", "0").lower()
            in {"1", "true", "yes"},
            task_progress_object_names=os.environ.get("ANTI_STUCK_TASK_PROGRESS_OBJECTS", ""),
            task_progress_distance=float(os.environ.get("ANTI_STUCK_TASK_PROGRESS_DISTANCE", "1.20")),
            task_progress_sticky_steps=int(os.environ.get("ANTI_STUCK_TASK_PROGRESS_STICKY_STEPS", "300")),
            log_prefix=log_prefix,
        )

    def reset(self) -> None:
        self.position_history.clear()
        self.action_history.clear()
        self.base_action_history.clear()
        self.nonbase_action_history.clear()
        self.eef_position_history.clear()
        self.gripper_position_history.clear()
        self.candidate_counter = 0
        self.idle_candidate_counter = 0
        self.escape_counter = 0
        self.cooldown_counter = self.initial_cooldown_steps
        self.escape_side = 1
        self.current_escape_scale = 1.0
        self.escape_start_xy = None
        self.num_triggers = 0
        self.max_trigger_logged = False
        self.finished_escape = False
        self.escape_stop_reason = ""
        self.escape_phase = "idle"
        self.escape_phase_step = 0
        self.escape_phase_start_xy = None
        self.probe_cycle = 0
        self.strafe_cycle = 0
        self.forward_cycle = 0
        self.commit_start_xy = None
        self.env_step_counter = 0
        self.step_counter = 0
        self.pending_stuck_reason = ""
        self.video_label = ""
        self.suppress_video_label = ""
        self.suppress_label_counter = 0
        self.escape_original_trunk_qpos = None
        self.last_escape_base_cmd = np.zeros(3, dtype=np.float64)
        self.tilt_recovery_attempts = 0
        self.tilt_resume_phase = ""
        self.tilt_recovery_cmd = np.zeros(3, dtype=np.float64)
        self.direction_start_yaw = 0.0
        self.direction_primary_side = 1
        self.direction_secondary_side = -1
        self.direction_chosen_side = 0
        self.direction_valid_side = True
        self.direction_side_switches = 0
        self.direction_probe_scores: dict[int, float] = {}
        self.direction_probe_lateral_progress: dict[int, float] = {}
        self.direction_probe_forward_progress: dict[int, float] = {}
        self.direction_tilt_blocked_sides: set[int] = set()
        self.task_progress_sticky_counter = 0
        self.task_progress_last_object = ""
        self.task_progress_last_distance = float("inf")
        self.task_progress_was_active = False

    def tick_env_step(self) -> None:
        self.env_step_counter += 1

    def _log(self, msg: str, *args: Any) -> None:
        text = msg % args if args else msg
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        env_step = getattr(self, "env_step_counter", -1)
        check_step = getattr(self, "step_counter", -1)
        print(f"[AntiStuck] {prefix}env_step={env_step} check_step={check_step} {text}", flush=True)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _parse_vector(value: Any, default: np.ndarray) -> np.ndarray:
        if value is None:
            return np.array(default, dtype=np.float64)
        if isinstance(value, str):
            pieces = [x.strip() for x in value.split(",") if x.strip()]
            try:
                parsed = np.array([float(x) for x in pieces], dtype=np.float64)
            except ValueError:
                return np.array(default, dtype=np.float64)
        else:
            parsed = np.asarray(value, dtype=np.float64).reshape(-1)
        if parsed.shape[0] != default.shape[0]:
            return np.array(default, dtype=np.float64)
        return parsed

    @staticmethod
    def _replace_base_action(template: Any, vx: float, vy: float, wz: float) -> torch.Tensor:
        if isinstance(template, torch.Tensor):
            action = template.clone()
            flat = action.reshape(-1)
            flat[0] = vx
            flat[1] = vy
            flat[2] = wz
            return action
        action_np = np.array(template, copy=True)
        flat = action_np.reshape(-1)
        flat[0] = vx
        flat[1] = vy
        flat[2] = wz
        return torch.as_tensor(action_np, dtype=torch.float32)

    @staticmethod
    def _replace_base_and_trunk_action(
        template: Any,
        vx: float,
        vy: float,
        wz: float,
        trunk_target: np.ndarray | None = None,
    ) -> torch.Tensor:
        if isinstance(template, torch.Tensor):
            action = template.clone()
            flat = action.reshape(-1)
            flat[0] = vx
            flat[1] = vy
            flat[2] = wz
            if trunk_target is not None and flat.numel() >= 7:
                flat[3:7] = torch.as_tensor(trunk_target, dtype=flat.dtype, device=flat.device)
            return action
        action_np = np.array(template, copy=True)
        flat = action_np.reshape(-1)
        flat[0] = vx
        flat[1] = vy
        flat[2] = wz
        if trunk_target is not None and flat.size >= 7:
            flat[3:7] = trunk_target
        return torch.as_tensor(action_np, dtype=torch.float32)

    @classmethod
    def _robot_xy(cls, robot: Any) -> np.ndarray:
        pos = cls._to_numpy(robot.get_position_orientation()[0]).reshape(-1)
        return pos[:2].astype(np.float64)

    @classmethod
    def _robot_trunk_qpos(cls, robot: Any) -> np.ndarray | None:
        if robot is None or not hasattr(robot, "trunk_control_idx"):
            return None
        try:
            joint_positions = cls._to_numpy(robot.get_joint_positions()).reshape(-1)
            trunk_idx = cls._to_numpy(robot.trunk_control_idx).reshape(-1).astype(int)
            return joint_positions[trunk_idx[:4]].astype(np.float64)
        except Exception:
            return None

    @classmethod
    def robot_pose(cls, robot: Any) -> tuple[np.ndarray, np.ndarray]:
        pos, orn = robot.get_position_orientation()
        return cls._to_numpy(pos).reshape(-1).astype(np.float64), cls._to_numpy(orn).reshape(-1).astype(np.float64)

    @classmethod
    def robot_rpy(cls, robot: Any) -> np.ndarray:
        from omnigibson.utils import transform_utils as T

        _pos, orn = robot.get_position_orientation()
        orn_tensor = orn if isinstance(orn, torch.Tensor) else torch.as_tensor(orn, dtype=torch.float32)
        rpy = T.quat2euler(orn_tensor.reshape(-1)[:4])
        return cls._to_numpy(rpy).reshape(-1).astype(np.float64)

    def _tilt_abs(self, robot: Any | None) -> tuple[float, float, float]:
        if robot is None:
            return 0.0, 0.0, 0.0
        try:
            roll, pitch, _yaw = self.robot_rpy(robot)
            return max(abs(float(roll)), abs(float(pitch))), float(roll), float(pitch)
        except Exception:
            return 0.0, 0.0, 0.0

    @classmethod
    def _robot_yaw(cls, robot: Any | None) -> float:
        if robot is None:
            return 0.0
        try:
            return float(cls.robot_rpy(robot)[2])
        except Exception:
            return 0.0

    @classmethod
    def _object_xy(cls, obj: Any) -> np.ndarray | None:
        try:
            pos = cls._to_numpy(obj.get_position_orientation()[0]).reshape(-1)
        except Exception:
            return None
        if pos.shape[0] < 2:
            return None
        return pos[:2].astype(np.float64)

    def _nearest_task_progress_object(
        self,
        robot: Any | None,
        task_scope: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        if robot is None or not self.task_progress_object_names or not task_scope:
            return "", float("inf")
        try:
            robot_xy = self._robot_xy(robot)
        except Exception:
            return "", float("inf")

        best_name = ""
        best_distance = float("inf")
        for name in self.task_progress_object_names:
            obj = task_scope.get(name)
            if obj is None or not getattr(obj, "exists", True) or getattr(obj, "is_system", False):
                continue
            obj_xy = self._object_xy(obj)
            if obj_xy is None:
                continue
            distance = float(np.linalg.norm(robot_xy - obj_xy))
            if distance < best_distance:
                best_name = name
                best_distance = distance
        return best_name, best_distance

    def _task_progress_guard_active(
        self,
        robot: Any | None,
        task_scope: dict[str, Any] | None = None,
    ) -> tuple[bool, str, float, bool]:
        if not self.task_progress_guard_enable:
            return False, "", float("inf"), False

        name, distance = self._nearest_task_progress_object(robot, task_scope)
        near_target = bool(name) and distance <= self.task_progress_distance
        if near_target:
            self.task_progress_sticky_counter = max(0, self.task_progress_sticky_steps)
            self.task_progress_last_object = name
            self.task_progress_last_distance = distance
            return True, name, distance, False

        if self.task_progress_sticky_counter > 0:
            self.task_progress_sticky_counter -= 1
            name = self.task_progress_last_object
            distance = self.task_progress_last_distance
            return True, name, distance, True

        self.task_progress_last_object = ""
        self.task_progress_last_distance = float("inf")
        return False, "", float("inf"), False

    def _suppress_for_task_progress(
        self,
        robot: Any | None,
        task_scope: dict[str, Any] | None = None,
        abort_escape: bool = False,
    ) -> bool:
        active, name, distance, sticky = self._task_progress_guard_active(robot, task_scope)
        if not active:
            if self.task_progress_was_active:
                self._log("version=%s, task progress guard cleared", self.version)
            self.task_progress_was_active = False
            return False

        sticky_text = " sticky" if sticky else ""
        distance_text = "inf" if not np.isfinite(distance) else f"{distance:.3f}"
        self.suppress_video_label = (
            f"{self.version} SUPPRESS near_egg{sticky_text} {name} d={distance_text}"
        )
        self.suppress_label_counter = 30
        should_log = (
            not self.task_progress_was_active
            or abort_escape
            or self.candidate_counter > 0
            or self.idle_candidate_counter > 0
            or self.step_counter % 1000 < 2
        )
        if abort_escape:
            self._finish_escape("task_progress_guard")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop task_progress"
        if should_log:
            self._log(
                "version=%s, suppressed by task progress guard: object=%s, distance=%s, "
                "threshold=%.3f, sticky=%s, abort_escape=%s",
                self.version,
                name or "unknown",
                distance_text,
                self.task_progress_distance,
                sticky,
                abort_escape,
            )
        self.task_progress_was_active = True
        self.candidate_counter = 0
        self.idle_candidate_counter = 0
        return True

    def suppress_for_task_progress(
        self,
        robot: Any | None,
        task_scope: dict[str, Any] | None = None,
        abort_escape: bool = False,
    ) -> bool:
        return self._suppress_for_task_progress(robot, task_scope, abort_escape=abort_escape)

    def _direction_axes(self) -> tuple[np.ndarray, np.ndarray]:
        yaw = float(self.direction_start_yaw)
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        return forward, left

    def _axis_progress(
        self,
        robot: Any | None,
        axis: np.ndarray,
        start_xy: np.ndarray | None = None,
    ) -> float | None:
        if robot is None:
            return None
        start = self.escape_start_xy if start_xy is None else start_xy
        if start is None:
            return None
        return float(np.dot(self._robot_xy(robot) - start, axis))

    def _lateral_progress_from_start(self, robot: Any | None, side: int) -> float | None:
        _forward, left = self._direction_axes()
        return self._axis_progress(robot, float(side) * left, self.escape_start_xy)

    def _lateral_progress_from_phase(self, robot: Any | None, side: int) -> float | None:
        _forward, left = self._direction_axes()
        return self._axis_progress(robot, float(side) * left, self.escape_phase_start_xy)

    def _forward_progress_from_phase(self, robot: Any | None) -> float | None:
        forward, _left = self._direction_axes()
        return self._axis_progress(robot, forward, self.escape_phase_start_xy)

    def _forward_progress_from_start(self, robot: Any | None) -> float | None:
        forward, _left = self._direction_axes()
        return self._axis_progress(robot, forward, self.escape_start_xy)

    def _backward_progress_from_phase(self, robot: Any | None) -> float | None:
        forward, _left = self._direction_axes()
        return self._axis_progress(robot, -forward, self.escape_phase_start_xy)

    def _record_direction_probe_score(self, side: int, robot: Any | None, tilt: float) -> None:
        forward, left = self._direction_axes()
        lateral_progress = self._axis_progress(robot, float(side) * left, self.escape_phase_start_xy)
        forward_progress = self._axis_progress(robot, forward, self.escape_phase_start_xy)
        lateral = 0.0 if lateral_progress is None else lateral_progress
        forward_delta = 0.0 if forward_progress is None else forward_progress
        score = lateral + 0.10 * max(0.0, forward_delta) - 0.25 * max(0.0, tilt - self.tilt_clear_rad)
        self.direction_probe_scores[int(side)] = float(score)
        self.direction_probe_lateral_progress[int(side)] = float(lateral)
        self.direction_probe_forward_progress[int(side)] = float(forward_delta)
        self._log(
            "version=%s, directional probe scored: side=%d, score=%.4f, lateral=%.4f, forward=%.4f, tilt=%.3f",
            self.version,
            side,
            score,
            lateral,
            forward_delta,
            tilt,
        )

    def _mark_direction_tilt_blocked(self, side: int, robot: Any | None, tilt: float, phase: str) -> None:
        side = int(side)
        forward, left = self._direction_axes()
        lateral_progress = self._axis_progress(robot, float(side) * left, self.escape_phase_start_xy)
        forward_progress = self._axis_progress(robot, forward, self.escape_phase_start_xy)
        lateral = 0.0 if lateral_progress is None else lateral_progress
        forward_delta = 0.0 if forward_progress is None else forward_progress
        self.direction_tilt_blocked_sides.add(side)
        self.direction_probe_scores[side] = -1e9
        self.direction_probe_lateral_progress[side] = 0.0
        self.direction_probe_forward_progress[side] = float(forward_delta)
        self._log(
            "version=%s, direction tilt-blocked: phase=%s, side=%d, lateral=%.4f, forward=%.4f, "
            "tilt=%.3f, blocked_sides=%s",
            self.version,
            phase,
            side,
            lateral,
            forward_delta,
            tilt,
            sorted(self.direction_tilt_blocked_sides),
        )

    def _choose_direction_side(self) -> int:
        primary = int(self.direction_primary_side)
        secondary = int(self.direction_secondary_side)
        primary_score = self.direction_probe_scores.get(primary, -1e9)
        secondary_score = self.direction_probe_scores.get(secondary, -1e9)
        primary_lateral = self.direction_probe_lateral_progress.get(primary, 0.0)
        secondary_lateral = self.direction_probe_lateral_progress.get(secondary, 0.0)
        blocked_sides = getattr(self, "direction_tilt_blocked_sides", set())

        primary_good = primary not in blocked_sides and primary_lateral >= self.direction_min_probe_progress
        secondary_good = secondary not in blocked_sides and secondary_lateral >= self.direction_min_probe_progress
        self.direction_valid_side = bool(primary_good or secondary_good)
        if primary_good and not secondary_good:
            chosen = primary
        elif secondary_good and not primary_good:
            chosen = secondary
        elif secondary_score > primary_score:
            chosen = secondary
        else:
            chosen = primary

        self.direction_chosen_side = int(chosen)
        self.escape_side = int(chosen)
        self._log(
            "version=%s, directional side chosen: chosen=%d, primary=%d score=%.4f lateral=%.4f, "
            "secondary=%d score=%.4f lateral=%.4f, min_probe_progress=%.4f, blocked_sides=%s, valid_side=%s",
            self.version,
            chosen,
            primary,
            primary_score,
            primary_lateral,
            secondary,
            secondary_score,
            secondary_lateral,
            self.direction_min_probe_progress,
            sorted(blocked_sides),
            self.direction_valid_side,
        )
        return int(chosen)

    @classmethod
    def _eef_positions(cls, robot: Any) -> np.ndarray:
        positions = []
        for arm in getattr(robot, "arm_names", []):
            positions.append(cls._to_numpy(robot.eef_links[arm].get_position_orientation()[0]).reshape(-1)[:3])
        if not positions:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(positions, axis=0).astype(np.float64)

    @classmethod
    def _gripper_positions(cls, robot: Any) -> np.ndarray:
        if not hasattr(robot, "gripper_control_idx"):
            return np.zeros(0, dtype=np.float64)
        joint_positions = cls._to_numpy(robot.get_joint_positions()).reshape(-1)
        values = []
        for arm in getattr(robot, "arm_names", []):
            values.append(joint_positions[np.asarray(robot.gripper_control_idx[arm], dtype=int)])
        if not values:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(values, axis=0).astype(np.float64)

    def _record_action(self, policy_action: Any) -> np.ndarray:
        flat_action = self._to_numpy(policy_action).reshape(-1).astype(np.float64)
        self.action_history.append(flat_action.copy())
        base = flat_action[:3]
        self.base_action_history.append(base.copy())
        self.nonbase_action_history.append(flat_action[3:].copy())
        return base

    def _record_manipulation_state(self, robot: Any) -> None:
        self.eef_position_history.append(self._eef_positions(robot))
        self.gripper_position_history.append(self._gripper_positions(robot))

    def _manipulation_guard_active(self) -> tuple[bool, float, float]:
        if self.manipulation_guard_window <= 1:
            return False, 0.0, 0.0
        recent_eef = list(self.eef_position_history)[-self.manipulation_guard_window :]
        if len(recent_eef) >= 2 and recent_eef[-1].size > 0:
            eef_recent = np.stack(recent_eef, axis=0)
            eef_deltas = np.linalg.norm(np.diff(eef_recent, axis=0), axis=1)
            eef_delta_ratio = float(np.mean(eef_deltas > self.manipulation_eef_delta_threshold))
        else:
            eef_delta_ratio = 0.0
        recent_gripper = list(self.gripper_position_history)[-self.manipulation_guard_window :]
        if len(recent_gripper) >= 2 and recent_gripper[-1].size > 0:
            gripper_recent = np.stack(recent_gripper, axis=0)
            gripper_deltas = np.linalg.norm(np.diff(gripper_recent, axis=0), axis=1)
            gripper_delta_ratio = float(np.mean(gripper_deltas > self.manipulation_gripper_delta_threshold))
        else:
            gripper_delta_ratio = 0.0
        gripper_active = gripper_delta_ratio >= self.manipulation_gripper_delta_ratio_threshold
        eef_and_gripper_active = eef_delta_ratio >= self.manipulation_eef_delta_ratio_threshold and gripper_active
        eef_only_extreme_active = eef_delta_ratio >= self.manipulation_eef_only_ratio_threshold
        return gripper_active or eef_and_gripper_active or eef_only_extreme_active, eef_delta_ratio, gripper_delta_ratio

    def _arm_deployed_guard_active(self) -> tuple[bool, float, float, float]:
        if self.arm_deployed_guard_window <= 1:
            return False, 0.0, 0.0, 0.0
        if len(self.action_history) < self.arm_deployed_guard_window:
            return False, 0.0, 0.0, 0.0

        recent = np.stack(list(self.action_history)[-self.arm_deployed_guard_window :], axis=0)
        if recent.shape[1] < 22:
            return False, 0.0, 0.0, 0.0

        base_norms = np.linalg.norm(recent[:, :2], axis=1)
        left_arm_abs = np.mean(np.abs(recent[:, 7:14]), axis=1)
        right_arm_abs = np.mean(np.abs(recent[:, 15:22]), axis=1)
        arm_abs = np.maximum(left_arm_abs, right_arm_abs)
        arm_abs_mean = float(np.mean(arm_abs))
        arm_deployed_ratio = float(np.mean(arm_abs >= self.arm_deployed_abs_threshold))
        base_norm_mean = float(np.mean(base_norms))
        active = (
            arm_deployed_ratio >= self.arm_deployed_ratio_threshold
            and base_norm_mean <= self.arm_deployed_base_command_threshold
        )
        return active, arm_abs_mean, arm_deployed_ratio, base_norm_mean

    def _base_window_stats(self) -> tuple[float, float, bool, float]:
        recent = np.stack(list(self.base_action_history)[-self.position_window :], axis=0)
        translation = recent[:, :2]
        norms = np.linalg.norm(translation, axis=1)
        command_norm = float(norms[-1])
        intent_ratio = float(np.mean(norms > self.base_command_threshold))
        repeated_translation = False
        repeat_std = float("nan")
        if len(self.base_action_history) >= self.repeat_window:
            repeat_recent = np.stack(list(self.base_action_history)[-self.repeat_window :], axis=0)
            repeat_std = float(np.max(np.std(repeat_recent[:, :2], axis=0)))
            repeated_translation = (
                repeat_std < self.repeat_std_threshold
                and float(np.linalg.norm(np.mean(repeat_recent[:, :2], axis=0))) > self.base_command_threshold
            )
        return command_norm, intent_ratio, repeated_translation, repeat_std

    def _policy_trunk_target(self, policy_action: Any) -> np.ndarray | None:
        flat = self._to_numpy(policy_action).reshape(-1).astype(np.float64)
        if flat.shape[0] < 7:
            return None
        return flat[3:7].copy()

    def _capture_original_trunk_qpos(self, robot: Any | None, policy_action: Any | None = None) -> np.ndarray | None:
        trunk_qpos = self._robot_trunk_qpos(robot)
        if trunk_qpos is not None and trunk_qpos.shape[0] == 4:
            return trunk_qpos
        if policy_action is not None:
            return self._policy_trunk_target(policy_action)
        return None

    def _crouch_target_for_action(self) -> np.ndarray | None:
        return self.crouch_target.copy() if self.crouch_enable else None

    def _crouch_interpolation_target(self) -> np.ndarray | None:
        if not self.crouch_enable:
            return None
        if self.escape_original_trunk_qpos is None:
            return self.crouch_target.copy()
        steps = max(1, self.crouch_steps)
        alpha = min(1.0, max(0.0, self.escape_phase_step / steps))
        return (1.0 - alpha) * self.escape_original_trunk_qpos + alpha * self.crouch_target

    def _escape_action(
        self,
        policy_action: Any,
        vx: float,
        vy: float,
        wz: float = 0.0,
        trunk_target: np.ndarray | None = None,
        remember_cmd: bool = True,
    ) -> torch.Tensor:
        if remember_cmd:
            self.last_escape_base_cmd = np.array([vx, vy, wz], dtype=np.float64)
        return self._replace_base_and_trunk_action(policy_action, vx, vy, wz, trunk_target)

    def update_and_check_stuck(
        self,
        robot: Any,
        policy_action: Any,
        task_scope: dict[str, Any] | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        self.step_counter += 1
        self.video_label = ""
        if self.suppress_label_counter > 0:
            self.suppress_label_counter -= 1
        else:
            self.suppress_video_label = ""
        self.position_history.append(self._robot_xy(robot).copy())
        self._record_action(policy_action)
        self._record_manipulation_state(robot)

        if self._suppress_for_task_progress(robot, task_scope):
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            return False

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.candidate_counter = 0
            self.idle_candidate_counter = 0
            return False
        if self.max_triggers > 0 and self.num_triggers >= self.max_triggers:
            if not self.max_trigger_logged:
                self._log("max_triggers=%d reached; disabled for this episode", self.max_triggers)
                self.max_trigger_logged = True
            return False

        arm_deployed_active, arm_abs_mean, arm_deployed_ratio, arm_base_norm_mean = self._arm_deployed_guard_active()
        if arm_deployed_active:
            self.suppress_video_label = (
                f"{self.version} SUPPRESS arm={arm_abs_mean:.3f} ratio={arm_deployed_ratio:.2f} "
                f"base={arm_base_norm_mean:.3f}"
            )
            self.suppress_label_counter = 30
            if (self.candidate_counter or self.idle_candidate_counter) or self.step_counter % 1000 < 2:
                self._log(
                    "version=%s, suppressed because arm is deployed: arm_abs_mean=%.3f threshold=%.3f, "
                    "arm_deployed_ratio=%.3f threshold=%.3f, base_norm_mean=%.3f threshold=%.3f",
                    self.version,
                    arm_abs_mean,
                    self.arm_deployed_abs_threshold,
                    arm_deployed_ratio,
                    self.arm_deployed_ratio_threshold,
                    arm_base_norm_mean,
                    self.arm_deployed_base_command_threshold,
                )
            self.candidate_counter = 0
            self.idle_candidate_counter = 0
            return False

        manipulation_active, eef_delta_ratio, gripper_delta_ratio = self._manipulation_guard_active()
        if manipulation_active:
            self.suppress_video_label = (
                f"{self.version} SUPPRESS eef={eef_delta_ratio:.3f}>={self.manipulation_eef_delta_ratio_threshold:.3f} "
                f"grip={gripper_delta_ratio:.3f}>={self.manipulation_gripper_delta_ratio_threshold:.3f}"
            )
            self.suppress_label_counter = 30
            if (self.candidate_counter or self.idle_candidate_counter) or self.step_counter % 1000 < 2:
                self._log(
                    "version=%s, suppressed during manipulation: eef_delta_ratio=%.3f threshold=%.3f, "
                    "gripper_delta_ratio=%.3f threshold=%.3f",
                    self.version,
                    eef_delta_ratio,
                    self.manipulation_eef_delta_ratio_threshold,
                    gripper_delta_ratio,
                    self.manipulation_gripper_delta_ratio_threshold,
                )
            self.candidate_counter = 0
            self.idle_candidate_counter = 0
            return False

        intent_ready = len(self.position_history) >= self.position_window and len(self.base_action_history) >= self.position_window
        idle_ready = (
            self.idle_position_window > 0
            and len(self.position_history) >= self.idle_position_window
            and len(self.base_action_history) >= self.idle_position_window
        )
        if intent_ready:
            positions = np.stack(list(self.position_history)[-self.position_window :], axis=0)
            xy_span = float(np.linalg.norm(np.max(positions, axis=0) - np.min(positions, axis=0)))
            command_norm, intent_ratio, repeated_translation, repeat_std = self._base_window_stats()
            has_current_base_intent = command_norm > self.base_command_threshold
            has_sustained_base_intent = intent_ratio >= self.intent_ratio_threshold and has_current_base_intent
            has_repeated_base_intent = repeated_translation and has_current_base_intent
            strong_position_threshold = self.position_threshold * self.strong_position_scale
            has_strong_static_base_intent = (
                command_norm >= self.strong_base_command_threshold and xy_span < strong_position_threshold
            )
            low_progress_xy_threshold = max(
                self.position_threshold,
                min(self.low_progress_position_threshold, command_norm * self.low_progress_position_scale),
            )
            has_low_progress_base_intent = (
                command_norm >= self.low_progress_command_threshold and xy_span < low_progress_xy_threshold
            )
            if (
                (xy_span < self.position_threshold and (
                    has_sustained_base_intent or has_repeated_base_intent or has_strong_static_base_intent
                ))
                or has_low_progress_base_intent
            ):
                self.candidate_counter += 1
                if self.candidate_counter >= self.confirm_steps:
                    if has_low_progress_base_intent and not has_strong_static_base_intent:
                        self.pending_stuck_reason = "base-intent-low-progress"
                    elif has_strong_static_base_intent:
                        self.pending_stuck_reason = "base-intent-strong"
                    else:
                        self.pending_stuck_reason = "base-intent"
                    self._log(
                        "stuck detected: reason=%s, step=%d, xy_span=%.4f over %d steps, "
                        "base_command_norm=%.4f, intent_ratio=%.3f, confirm_steps=%d, "
                        "repeated_translation=%s, repeat_std=%.5f, strong_static=%s, low_progress=%s, "
                        "low_progress_xy_threshold=%.4f",
                        self.pending_stuck_reason,
                        self.step_counter,
                        xy_span,
                        self.position_window,
                        command_norm,
                        intent_ratio,
                        self.candidate_counter,
                        repeated_translation,
                        repeat_std,
                        has_strong_static_base_intent,
                        has_low_progress_base_intent,
                        low_progress_xy_threshold,
                    )
                    return True
            else:
                self.candidate_counter = 0
        else:
            self.candidate_counter = 0

        if idle_ready:
            idle_positions = np.stack(list(self.position_history)[-self.idle_position_window :], axis=0)
            idle_xy_span = float(np.linalg.norm(np.max(idle_positions, axis=0) - np.min(idle_positions, axis=0)))
            idle_actions = np.stack(list(self.base_action_history)[-self.idle_position_window :], axis=0)
            idle_norms = np.linalg.norm(idle_actions[:, :2], axis=1)
            idle_intent_ratio = float(np.mean(idle_norms > self.base_command_threshold))
            command_norm = float(idle_norms[-1])
            nonbase_actions = np.stack(list(self.nonbase_action_history)[-self.idle_position_window :], axis=0)
            if len(nonbase_actions) > 1 and nonbase_actions.shape[1] > 0:
                nonbase_deltas = np.linalg.norm(np.diff(nonbase_actions, axis=0), axis=1)
                nonbase_delta_ratio = float(np.mean(nonbase_deltas > self.idle_nonbase_delta_threshold))
            else:
                nonbase_delta_ratio = 0.0
            if (
                idle_xy_span < self.idle_position_threshold
                and idle_intent_ratio <= self.idle_intent_ratio_threshold
                and nonbase_delta_ratio <= self.idle_nonbase_delta_ratio_threshold
            ):
                self.idle_candidate_counter += 1
                if self.idle_candidate_counter >= self.idle_confirm_steps:
                    self.pending_stuck_reason = "idle-static"
                    self._log(
                        "stuck detected: reason=%s, step=%d, xy_span=%.4f over %d steps, "
                        "base_command_norm=%.4f, idle_intent_ratio=%.3f threshold=%.3f, "
                        "nonbase_delta_ratio=%.3f threshold=%.3f, confirm_steps=%d",
                        self.pending_stuck_reason,
                        self.step_counter,
                        idle_xy_span,
                        self.idle_position_window,
                        command_norm,
                        idle_intent_ratio,
                        self.idle_intent_ratio_threshold,
                        nonbase_delta_ratio,
                        self.idle_nonbase_delta_ratio_threshold,
                        self.idle_candidate_counter,
                    )
                    return True
            else:
                self.idle_candidate_counter = 0
        else:
            self.idle_candidate_counter = 0
        return False

    def start_escape(self, robot: Any | None = None) -> None:
        self.candidate_counter = 0
        self.idle_candidate_counter = 0
        self.escape_counter = self.escape_steps
        self.escape_side = 1 if np.random.random() < 0.5 else -1
        if self.escape_mode == "retreat_left_gap_commit":
            self.escape_side = 1
        self.num_triggers += 1
        self.current_escape_scale = 1.0 + max(0, self.num_triggers - 1) * self.escape_scale_growth
        self.escape_start_xy = self._robot_xy(robot) if robot is not None else None
        self.direction_start_yaw = self._robot_yaw(robot)
        self.escape_original_trunk_qpos = self._capture_original_trunk_qpos(robot)
        self.finished_escape = False
        self.escape_stop_reason = ""
        self.direction_primary_side = int(self.escape_side)
        self.direction_secondary_side = int(-self.escape_side)
        self.direction_chosen_side = 0
        self.direction_valid_side = True
        self.direction_side_switches = 0
        self.direction_probe_scores = {}
        self.direction_probe_lateral_progress = {}
        self.direction_probe_forward_progress = {}
        self.direction_tilt_blocked_sides = set()
        if self.escape_mode in {
            "probe_commit",
            "forward_strafe_retry",
            "fixed_forward_strafe",
            "directional_sidestep_commit",
            "retreat_left_gap_commit",
            "conditional_retreat_left_gap",
        }:
            if self.crouch_enable and self.crouch_steps > 0:
                self.escape_phase = "pre_crouch"
            elif self.escape_mode == "probe_commit":
                self.escape_phase = "strafe_probe"
            elif self.escape_mode == "directional_sidestep_commit":
                self.escape_phase = "direction_probe_primary"
            elif self.escape_mode == "retreat_left_gap_commit":
                self.escape_phase = "retreat_back"
            elif self.escape_mode == "conditional_retreat_left_gap":
                self.escape_phase = "conditional_forward_probe"
            else:
                self.escape_phase = "forward_try"
        else:
            self.escape_phase = "legacy"
        self.escape_phase_step = 0
        self.escape_phase_start_xy = None if self.escape_start_xy is None else self.escape_start_xy.copy()
        self.probe_cycle = 1
        self.strafe_cycle = 0
        self.forward_cycle = 1
        self.commit_start_xy = None
        self.last_escape_base_cmd = np.zeros(3, dtype=np.float64)
        self.tilt_recovery_attempts = 0
        self.tilt_resume_phase = ""
        self.tilt_recovery_cmd = np.zeros(3, dtype=np.float64)
        side_label = "left" if self.escape_side > 0 else "right"
        self._log(
            "version=%s, start escape #%d: reason=%s, side=%d, side_label=%s, mode=%s, "
            "steps=%d, scale=%.2f, start_xy=%s, start_yaw=%.3f, trunk_qpos=%s, crouch_target=%s",
            self.version,
            self.num_triggers,
            self.pending_stuck_reason or "unknown",
            self.escape_side,
            side_label,
            self.escape_mode,
            self.escape_steps,
            self.current_escape_scale,
            np.round(self.escape_start_xy, 4).tolist() if self.escape_start_xy is not None else None,
            self.direction_start_yaw,
            np.round(self.escape_original_trunk_qpos, 4).tolist()
            if self.escape_original_trunk_qpos is not None
            else None,
            np.round(self.crouch_target, 4).tolist() if self.crouch_enable else None,
        )

    def is_escaping(self) -> bool:
        return self.enabled and self.escape_counter > 0

    def _current_escape_displacement(self, robot: Any | None = None) -> float | None:
        if robot is None or self.escape_start_xy is None:
            return None
        return float(np.linalg.norm(self._robot_xy(robot) - self.escape_start_xy))

    def _phase_displacement(self, robot: Any | None = None) -> float | None:
        if robot is None or self.escape_phase_start_xy is None:
            return None
        return float(np.linalg.norm(self._robot_xy(robot) - self.escape_phase_start_xy))

    def _start_escape_phase(self, phase: str, robot: Any | None = None) -> None:
        self.escape_phase = phase
        self.escape_phase_step = 0
        self.escape_phase_start_xy = self._robot_xy(robot) if robot is not None else None
        if phase in {"forward_commit", "direction_commit_forward"}:
            self.commit_start_xy = None if self.escape_phase_start_xy is None else self.escape_phase_start_xy.copy()
        self._log(
            "version=%s, escape phase=%s, probe_cycle=%d, strafe_cycle=%d, forward_cycle=%d, phase_start_xy=%s",
            self.version,
            phase,
            self.probe_cycle,
            self.strafe_cycle,
            self.forward_cycle,
            np.round(self.escape_phase_start_xy, 4).tolist() if self.escape_phase_start_xy is not None else None,
        )

    def _start_tilt_recovery(self, robot: Any | None, tilt: float, roll: float, pitch: float) -> None:
        self.tilt_recovery_attempts += 1
        resume_phase = self.escape_phase if self.escape_phase not in {"tilt_recover", "finished"} else self.tilt_resume_phase
        if self.escape_mode == "fixed_forward_strafe" and resume_phase == "forward_try":
            resume_phase = "strafe_step"
        self.tilt_resume_phase = resume_phase
        reverse_cmd = -self.last_escape_base_cmd * self.tilt_recovery_scale
        if np.linalg.norm(reverse_cmd[:2]) < 1e-5:
            reverse_cmd = np.array([self.back_vel * self.current_escape_scale, 0.0, 0.0], dtype=np.float64)
        reverse_cmd[2] = 0.0
        self.tilt_recovery_cmd = reverse_cmd
        self._start_escape_phase("tilt_recover", robot)
        self._log(
            "version=%s, tilt recovery start: attempt=%d/%d, tilt=%.3f, roll=%.3f, pitch=%.3f, "
            "resume_phase=%s, reverse_cmd=%s",
            self.version,
            self.tilt_recovery_attempts,
            self.tilt_recovery_max_attempts,
            tilt,
            roll,
            pitch,
            self.tilt_resume_phase or "unknown",
            np.round(self.tilt_recovery_cmd, 4).tolist(),
        )

    def _route_conditional_after_tilt_recovery(
        self,
        robot: Any | None,
        tilt: float,
        roll: float,
        pitch: float,
        reason: str,
    ) -> bool:
        resume_phase = self.tilt_resume_phase
        if not resume_phase:
            return False

        self._log(
            "version=%s, conditional_retreat tilt recovered; abandoning triggering phase: "
            "resume_phase=%s, reason=%s, attempts=%d, roll=%.3f, pitch=%.3f",
            self.version,
            resume_phase,
            reason,
            self.tilt_recovery_attempts,
            roll,
            pitch,
        )
        self.tilt_recovery_attempts = 0
        self.tilt_resume_phase = ""
        self.tilt_recovery_cmd = np.zeros(3, dtype=np.float64)

        if resume_phase == "conditional_forward_probe":
            self._log(
                "version=%s, conditional_retreat forward direction abandoned after tilt; probing laterals",
                self.version,
            )
            self._start_escape_phase("conditional_probe_primary", robot)
            return True

        if resume_phase == "conditional_probe_primary":
            side = int(self.direction_primary_side)
            self._mark_direction_tilt_blocked(side, robot, tilt, resume_phase)
            self._start_escape_phase("conditional_return_primary", robot)
            return True

        if resume_phase == "conditional_return_primary":
            side = int(self.direction_secondary_side)
            self._mark_direction_tilt_blocked(side, robot, tilt, resume_phase)
            self._choose_direction_side()
            if self.direction_valid_side:
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._start_escape_phase("conditional_commit_strafe", robot)
            else:
                self.escape_side = 1
                self.probe_cycle = max(1, self.probe_cycle)
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._log(
                    "version=%s, conditional_retreat no lateral side remains after return tilt; using retreat-left fallback",
                    self.version,
                )
                self._start_escape_phase("retreat_back", robot)
            return True

        if resume_phase == "conditional_probe_secondary":
            side = int(self.direction_secondary_side)
            self._mark_direction_tilt_blocked(side, robot, tilt, resume_phase)
            self._choose_direction_side()
            if self.direction_valid_side:
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._start_escape_phase("conditional_commit_strafe", robot)
            else:
                self.escape_side = 1
                self.probe_cycle = max(1, self.probe_cycle)
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._log(
                    "version=%s, conditional_retreat both lateral sides tilt-blocked; using retreat-left fallback",
                    self.version,
                )
                self._start_escape_phase("retreat_back", robot)
            return True

        if resume_phase == "conditional_commit_strafe":
            old_side = int(self.escape_side)
            new_side = int(-old_side)
            self._mark_direction_tilt_blocked(old_side, robot, tilt, resume_phase)
            if (
                new_side not in self.direction_tilt_blocked_sides
                and self.direction_side_switches < self.direction_max_side_switches
            ):
                self.direction_side_switches += 1
                self.escape_side = new_side
                self.direction_chosen_side = new_side
                self._log(
                    "version=%s, conditional_retreat tilt during strafe commit; switching side: "
                    "old_side=%d, new_side=%d, switches=%d/%d",
                    self.version,
                    old_side,
                    new_side,
                    self.direction_side_switches,
                    self.direction_max_side_switches,
                )
                self._start_escape_phase("conditional_commit_strafe", robot)
            else:
                self.escape_side = 1
                self.probe_cycle = max(1, self.probe_cycle)
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._log(
                    "version=%s, conditional_retreat strafe side tilt-blocked and no switch remains; "
                    "using retreat-left fallback: old_side=%d, blocked_sides=%s",
                    self.version,
                    old_side,
                    sorted(self.direction_tilt_blocked_sides),
                )
                self._start_escape_phase("retreat_back", robot)
            return True

        if resume_phase == "conditional_commit_forward":
            if self.strafe_cycle >= self.max_strafe_cycles:
                self.escape_side = 1
                self.probe_cycle = max(1, self.probe_cycle)
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._log(
                    "version=%s, conditional_retreat forward commit tilted after max strafe cycles; "
                    "using retreat-left fallback: strafe_cycle=%d/%d",
                    self.version,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("retreat_back", robot)
            else:
                self.forward_cycle += 1
                self._log(
                    "version=%s, conditional_retreat forward commit caused tilt; "
                    "continuing lateral search instead of retrying forward: side=%d, strafe_cycle=%d/%d",
                    self.version,
                    self.escape_side,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("conditional_commit_strafe", robot)
            return True

        if resume_phase == "retreat_back":
            self._log(
                "version=%s, conditional_retreat retreat-back caused tilt; trying fallback left strafe",
                self.version,
            )
            self._start_escape_phase("retreat_left_strafe", robot)
            return True

        if resume_phase == "retreat_left_strafe":
            if self.probe_cycle >= self.max_probe_cycles:
                self._finish_escape("conditional_left_strafe_tilt_blocked")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop left_strafe_tilt"
                self._log(
                    "version=%s, conditional_retreat fallback left strafe tilt-blocked: retreat_cycle=%d/%d",
                    self.version,
                    self.probe_cycle,
                    self.max_probe_cycles,
                )
            else:
                self.probe_cycle += 1
                self._log(
                    "version=%s, conditional_retreat fallback left strafe caused tilt; retreating farther: "
                    "retreat_cycle=%d/%d",
                    self.version,
                    self.probe_cycle,
                    self.max_probe_cycles,
                )
                self._start_escape_phase("retreat_back", robot)
            return True

        if resume_phase == "retreat_forward_commit":
            if self.strafe_cycle >= self.max_strafe_cycles:
                self._finish_escape("conditional_retreat_forward_tilt_blocked")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop fallback_forward_tilt"
                self._log(
                    "version=%s, conditional_retreat fallback forward tilt-blocked: strafe_cycle=%d/%d",
                    self.version,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
            else:
                self.forward_cycle += 1
                self._log(
                    "version=%s, conditional_retreat fallback forward caused tilt; continuing left strafe: "
                    "strafe_cycle=%d/%d",
                    self.version,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("retreat_left_strafe", robot)
            return True

        self._start_escape_phase(resume_phase, robot)
        return True

    def _finish_escape(self, reason: str) -> None:
        self.escape_counter = 0
        self.finished_escape = True
        self.escape_stop_reason = reason
        self.escape_phase = "finished"
        self.position_history.clear()
        self.action_history.clear()
        self.base_action_history.clear()
        self.nonbase_action_history.clear()
        self.eef_position_history.clear()
        self.gripper_position_history.clear()
        self.idle_candidate_counter = 0
        self.pending_stuck_reason = ""
        self.tilt_resume_phase = ""
        self.tilt_recovery_cmd = np.zeros(3, dtype=np.float64)

    def step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        if self.escape_mode == "probe_commit":
            return self._probe_commit_step(policy_action, robot)
        if self.escape_mode in {"forward_strafe_retry", "fixed_forward_strafe"}:
            return self._forward_strafe_retry_step(policy_action, robot)
        if self.escape_mode == "directional_sidestep_commit":
            return self._directional_sidestep_commit_step(policy_action, robot)
        if self.escape_mode == "retreat_left_gap_commit":
            return self._retreat_left_gap_commit_step(policy_action, robot)
        if self.escape_mode == "conditional_retreat_left_gap":
            return self._conditional_retreat_left_gap_step(policy_action, robot)
        return self._legacy_step(policy_action, robot)

    def _legacy_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        displacement = self._current_escape_displacement(robot)
        if self.escape_max_displacement > 0 and displacement is not None and displacement >= self.escape_max_displacement:
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, escape stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                displacement,
                self.escape_max_displacement,
            )
            return self._replace_base_action(policy_action, 0.0, 0.0, 0.0)

        self.escape_counter -= 1
        t = self.escape_steps - self.escape_counter
        lateral_steps = max(1, min(self.escape_steps - 1, int(round(self.escape_steps * self.escape_lateral_fraction))))
        q1 = max(1, int(round(lateral_steps * 0.5)))
        q2 = lateral_steps
        q3 = min(self.escape_steps, max(q2 + 1, int(round(q2 + (self.escape_steps - q2) * 0.5))))

        if t <= lateral_steps:
            side_label = "left" if self.escape_side > 0 else "right"
            mode = f"strafe_{side_label}"
            action = self._replace_base_action(policy_action, 0.0, self.escape_side * self.strafe_vel * self.current_escape_scale, 0.0)
        else:
            mode = "forward"
            action = self._replace_base_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0)

        if displacement is not None and self.escape_max_displacement > 0:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} {t}/{self.escape_steps} "
                f"d={displacement:.2f}/{self.escape_max_displacement:.2f}"
            )
        else:
            self.video_label = f"{self.version} AS#{self.num_triggers} {mode} {t}/{self.escape_steps}"
        if t in {1, q1, q2, q3}:
            self._log(
                "version=%s, escape step=%d/%d, mode=%s, displacement=%s",
                self.version,
                t,
                self.escape_steps,
                mode,
                None if displacement is None else round(displacement, 4),
            )
        if self.escape_counter <= 0:
            self._finish_escape("steps")
        return action

    def _probe_commit_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        total_displacement = self._current_escape_displacement(robot)
        if (
            self.escape_max_displacement > 0
            and total_displacement is not None
            and total_displacement >= self.escape_max_displacement
        ):
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={total_displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, probe_commit stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                total_displacement,
                self.escape_max_displacement,
            )
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        phase_displacement = self._phase_displacement(robot)
        tilt, roll, pitch = self._tilt_abs(robot)
        if (
            self.tilt_recovery_enable
            and self.escape_phase not in {"pre_crouch", "tilt_recover", "finished"}
            and tilt >= self.tilt_warn_rad
        ):
            if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                self._finish_escape("tilt_recover_failed")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                    f"r={roll:.2f} p={pitch:.2f}"
                )
                self._log(
                    "version=%s, tilt recovery failed before new attempt: attempts=%d, roll=%.3f, pitch=%.3f",
                    self.version,
                    self.tilt_recovery_attempts,
                    roll,
                    pitch,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            self._start_tilt_recovery(robot, tilt, roll, pitch)

        if self.escape_phase == "pre_crouch":
            if self.escape_phase_step >= self.crouch_steps:
                self._start_escape_phase("strafe_probe", robot)
                phase_displacement = 0.0
        elif self.escape_phase == "tilt_recover":
            if (
                self.escape_phase_step >= self.tilt_recovery_min_steps
                and tilt <= self.tilt_clear_rad
                and self.tilt_resume_phase
            ):
                self._start_escape_phase(self.tilt_resume_phase, robot)
                phase_displacement = 0.0
            elif self.escape_phase_step >= self.tilt_recovery_steps:
                if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                    self._finish_escape("tilt_recover_failed")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                        f"r={roll:.2f} p={pitch:.2f}"
                    )
                    self._log(
                        "version=%s, tilt recovery failed: attempts=%d, roll=%.3f, pitch=%.3f",
                        self.version,
                        self.tilt_recovery_attempts,
                        roll,
                        pitch,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_tilt_recovery(robot, tilt, roll, pitch)
                phase_displacement = 0.0
        elif self.escape_phase == "strafe_probe":
            if (
                phase_displacement is not None
                and phase_displacement >= self.lateral_probe_distance
            ) or self.escape_phase_step >= self.lateral_probe_steps:
                self._start_escape_phase("forward_probe", robot)
                phase_displacement = 0.0
        elif self.escape_phase == "forward_probe":
            if (
                phase_displacement is not None
                and phase_displacement >= self.forward_probe_distance
            ):
                self._start_escape_phase("forward_commit", robot)
                phase_displacement = 0.0
            elif self.escape_phase_step >= self.forward_probe_steps:
                if self.probe_cycle >= self.max_probe_cycles:
                    self._finish_escape("probe_exhausted")
                    self.video_label = f"{self.version} AS#{self.num_triggers} stop probe_exhausted"
                    self._log(
                        "version=%s, probe_commit exhausted: probe_cycle=%d, total_displacement=%s",
                        self.version,
                        self.probe_cycle,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.probe_cycle += 1
                self._start_escape_phase("strafe_probe", robot)
                phase_displacement = 0.0
        elif self.escape_phase == "forward_commit":
            if (
                self.forward_commit_stall_check_steps > 0
                and self.escape_phase_step >= self.forward_commit_stall_check_steps
                and phase_displacement is not None
                and phase_displacement < self.forward_commit_min_progress
            ):
                self._finish_escape("commit_stalled")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop commit_stalled "
                    f"d={phase_displacement:.2f}<{self.forward_commit_min_progress:.2f}"
                )
                self._log(
                    "version=%s, probe_commit stalled: commit_displacement=%.4f, "
                    "min_progress=%.4f, phase_step=%d, total_displacement=%s",
                    self.version,
                    phase_displacement,
                    self.forward_commit_min_progress,
                    self.escape_phase_step,
                    None if total_displacement is None else round(total_displacement, 4),
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if (
                phase_displacement is not None
                and phase_displacement >= self.forward_commit_distance
            ) or self.escape_phase_step >= self.forward_commit_steps:
                self._finish_escape("commit_done")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop commit_done"
                self._log(
                    "version=%s, probe_commit done: commit_displacement=%s, total_displacement=%s",
                    self.version,
                    None if phase_displacement is None else round(phase_displacement, 4),
                    None if total_displacement is None else round(total_displacement, 4),
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        self.escape_phase_step += 1
        side_label = "left" if self.escape_side > 0 else "right"
        phase_displacement = self._phase_displacement(robot)
        phase_displacement = 0.0 if phase_displacement is None else phase_displacement

        trunk_target = self._crouch_target_for_action()
        if self.escape_phase == "pre_crouch":
            mode = "pre_crouch"
            target = 1.0
            trunk_target = self._crouch_interpolation_target()
            action = self._escape_action(policy_action, 0.0, 0.0, 0.0, trunk_target, remember_cmd=False)
        elif self.escape_phase == "tilt_recover":
            mode = f"tilt_recover r={roll:.2f} p={pitch:.2f}"
            target = self.tilt_clear_rad
            vx, vy, wz = self.tilt_recovery_cmd.tolist()
            action = self._escape_action(policy_action, vx, vy, wz, trunk_target, remember_cmd=False)
        elif self.escape_phase == "strafe_probe":
            mode = f"strafe_probe {side_label}"
            target = self.lateral_probe_distance
            action = self._escape_action(
                policy_action,
                0.0,
                self.escape_side * self.strafe_vel * self.current_escape_scale,
                0.0,
                trunk_target,
            )
        elif self.escape_phase == "forward_probe":
            mode = "forward_probe"
            target = self.forward_probe_distance
            action = self._escape_action(
                policy_action,
                self.forward_vel * self.current_escape_scale,
                0.0,
                0.0,
                trunk_target,
            )
        elif self.escape_phase == "forward_commit":
            mode = "forward_commit"
            target = self.forward_commit_distance
            action = self._escape_action(
                policy_action,
                self.forward_vel * self.current_escape_scale,
                0.0,
                0.0,
                trunk_target,
            )
        else:
            self._finish_escape("unknown_phase")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop unknown_phase"
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        if self.escape_phase == "pre_crouch":
            progress = min(1.0, self.escape_phase_step / max(1, self.crouch_steps))
            self.video_label = f"{self.version} AS#{self.num_triggers} {mode} {progress:.2f}"
        elif self.escape_phase == "tilt_recover":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"{self.escape_phase_step}/{self.tilt_recovery_steps} clear={self.tilt_clear_rad:.2f}"
            )
        else:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"p{self.probe_cycle:02d} d={phase_displacement:.2f}/{target:.2f}"
            )
        if self.escape_phase_step in {1, max(1, int(target * 100)), 30, 60, 90, 120}:
            self._log(
                "version=%s, probe_commit step: phase=%s, probe_cycle=%d, phase_step=%d, "
                "phase_displacement=%.4f, target=%.4f, total_displacement=%s, roll=%.3f, pitch=%.3f",
                self.version,
                self.escape_phase,
                self.probe_cycle,
                self.escape_phase_step,
                phase_displacement,
                target,
                None if total_displacement is None else round(total_displacement, 4),
                roll,
                pitch,
            )
        return action

    def _forward_strafe_retry_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        fixed_mode = self.escape_mode == "fixed_forward_strafe"
        total_displacement = self._current_escape_displacement(robot)
        if (
            self.escape_max_displacement > 0
            and total_displacement is not None
            and total_displacement >= self.escape_max_displacement
        ):
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={total_displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, forward_strafe_retry stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                total_displacement,
                self.escape_max_displacement,
            )
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        phase_displacement = self._phase_displacement(robot)
        tilt, roll, pitch = self._tilt_abs(robot)
        if (
            self.tilt_recovery_enable
            and self.escape_phase not in {"pre_crouch", "tilt_recover", "finished"}
            and tilt >= self.tilt_warn_rad
        ):
            if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                self._finish_escape("tilt_recover_failed")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                    f"r={roll:.2f} p={pitch:.2f}"
                )
                self._log(
                    "version=%s, tilt recovery failed before new attempt: attempts=%d, roll=%.3f, pitch=%.3f",
                    self.version,
                    self.tilt_recovery_attempts,
                    roll,
                    pitch,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            self._start_tilt_recovery(robot, tilt, roll, pitch)

        if self.escape_phase == "pre_crouch":
            if self.escape_phase_step >= self.crouch_steps:
                self._start_escape_phase("forward_try", robot)
                phase_displacement = 0.0
        elif self.escape_phase == "tilt_recover":
            if (
                self.escape_phase_step >= self.tilt_recovery_min_steps
                and tilt <= self.tilt_clear_rad
                and self.tilt_resume_phase
            ):
                self._start_escape_phase(self.tilt_resume_phase, robot)
                phase_displacement = 0.0
            elif self.escape_phase_step >= self.tilt_recovery_steps:
                if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                    self._finish_escape("tilt_recover_failed")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                        f"r={roll:.2f} p={pitch:.2f}"
                    )
                    self._log(
                        "version=%s, tilt recovery failed: attempts=%d, roll=%.3f, pitch=%.3f",
                        self.version,
                        self.tilt_recovery_attempts,
                        roll,
                        pitch,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_tilt_recovery(robot, tilt, roll, pitch)
                phase_displacement = 0.0
        elif self.escape_phase == "forward_try":
            forward_reached_target = phase_displacement is not None and phase_displacement >= self.forward_try_distance
            forward_below_min_progress = (
                self.forward_stall_check_steps > 0
                and self.escape_phase_step >= self.forward_stall_check_steps
                and phase_displacement is not None
                and phase_displacement < self.forward_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.forward_try_steps
            if fixed_mode and (forward_reached_target or forward_below_min_progress or forward_timed_out):
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("strafe_exhausted")
                    self.video_label = f"{self.version} AS#{self.num_triggers} stop strafe_exhausted"
                    self._log(
                        "version=%s, fixed_forward_strafe exhausted before mandatory strafe: "
                        "strafe_cycle=%d, forward_displacement=%s, min_progress=%.4f, "
                        "target=%.4f, timed_out=%s, total_displacement=%s",
                        self.version,
                        self.strafe_cycle,
                        None if phase_displacement is None else round(phase_displacement, 4),
                        self.forward_stall_min_progress,
                        self.forward_try_distance,
                        forward_timed_out,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._log(
                    "version=%s, fixed_forward_strafe forcing mandatory strafe: "
                    "forward_displacement=%s, target=%.4f, min_progress=%.4f, "
                    "below_min_progress=%s, timed_out=%s",
                    self.version,
                    None if phase_displacement is None else round(phase_displacement, 4),
                    self.forward_try_distance,
                    self.forward_stall_min_progress,
                    forward_below_min_progress,
                    forward_timed_out,
                )
                self._start_escape_phase("strafe_step", robot)
                phase_displacement = 0.0
            elif (not fixed_mode) and forward_reached_target:
                self.forward_cycle += 1
                self._start_escape_phase("forward_try", robot)
                phase_displacement = 0.0
            elif (not fixed_mode) and forward_below_min_progress:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("strafe_exhausted")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop strafe_exhausted "
                        f"d={phase_displacement:.2f}<{self.forward_stall_min_progress:.2f}"
                    )
                    self._log(
                        "version=%s, forward_strafe_retry exhausted after stalled forward: "
                        "strafe_cycle=%d, forward_displacement=%.4f, min_progress=%.4f, total_displacement=%s",
                        self.version,
                        self.strafe_cycle,
                        phase_displacement,
                        self.forward_stall_min_progress,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_escape_phase("strafe_step", robot)
                phase_displacement = 0.0
            elif (not fixed_mode) and forward_timed_out:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("strafe_exhausted")
                    self.video_label = f"{self.version} AS#{self.num_triggers} stop strafe_exhausted"
                    self._log(
                        "version=%s, forward_strafe_retry exhausted after forward timeout: "
                        "strafe_cycle=%d, forward_displacement=%s, total_displacement=%s",
                        self.version,
                        self.strafe_cycle,
                        None if phase_displacement is None else round(phase_displacement, 4),
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_escape_phase("strafe_step", robot)
                phase_displacement = 0.0
        elif self.escape_phase == "strafe_step":
            if (
                phase_displacement is not None
                and phase_displacement >= self.strafe_step_distance
            ) or self.escape_phase_step >= self.strafe_step_steps:
                self.strafe_cycle += 1
                self.forward_cycle += 1
                self._start_escape_phase("forward_try", robot)
                phase_displacement = 0.0

        self.escape_phase_step += 1
        side_label = "left" if self.escape_side > 0 else "right"
        phase_displacement = self._phase_displacement(robot)
        phase_displacement = 0.0 if phase_displacement is None else phase_displacement

        trunk_target = self._crouch_target_for_action()
        if self.escape_phase == "pre_crouch":
            mode = "pre_crouch"
            target = 1.0
            trunk_target = self._crouch_interpolation_target()
            action = self._escape_action(policy_action, 0.0, 0.0, 0.0, trunk_target, remember_cmd=False)
        elif self.escape_phase == "tilt_recover":
            mode = f"tilt_recover r={roll:.2f} p={pitch:.2f}"
            target = self.tilt_clear_rad
            vx, vy, wz = self.tilt_recovery_cmd.tolist()
            action = self._escape_action(policy_action, vx, vy, wz, trunk_target, remember_cmd=False)
        elif self.escape_phase == "forward_try":
            mode = f"forward_try f{self.forward_cycle:02d}"
            target = self.forward_try_distance
            action = self._escape_action(
                policy_action,
                self.forward_vel * self.current_escape_scale,
                0.0,
                0.0,
                trunk_target,
            )
        elif self.escape_phase == "strafe_step":
            mode = f"strafe_step {side_label} s{self.strafe_cycle + 1:02d}"
            target = self.strafe_step_distance
            action = self._escape_action(
                policy_action,
                0.0,
                self.escape_side * self.strafe_vel * self.current_escape_scale,
                0.0,
                trunk_target,
            )
        else:
            self._finish_escape("unknown_phase")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop unknown_phase"
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        if self.escape_phase == "pre_crouch":
            progress = min(1.0, self.escape_phase_step / max(1, self.crouch_steps))
            self.video_label = f"{self.version} AS#{self.num_triggers} {mode} {progress:.2f}"
        elif self.escape_phase == "tilt_recover":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"{self.escape_phase_step}/{self.tilt_recovery_steps} clear={self.tilt_clear_rad:.2f}"
            )
        elif self.escape_phase == "forward_try":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"d={phase_displacement:.2f}/{target:.2f} strafe={self.strafe_cycle}/{self.max_strafe_cycles}"
            )
        else:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"d={phase_displacement:.2f}/{target:.2f} strafe={self.strafe_cycle}/{self.max_strafe_cycles}"
            )
        if self.escape_phase_step in {1, 30, 60, 90, 120}:
            self._log(
                "version=%s, forward_strafe_retry step: phase=%s, forward_cycle=%d, "
                "strafe_cycle=%d/%d, phase_step=%d, phase_displacement=%.4f, "
                "target=%.4f, total_displacement=%s, roll=%.3f, pitch=%.3f",
                self.version,
                self.escape_phase,
                self.forward_cycle,
                self.strafe_cycle,
                self.max_strafe_cycles,
                self.escape_phase_step,
                phase_displacement,
                target,
                None if total_displacement is None else round(total_displacement, 4),
                roll,
                pitch,
            )
        return action

    def _directional_sidestep_commit_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        total_displacement = self._current_escape_displacement(robot)
        if (
            self.escape_max_displacement > 0
            and total_displacement is not None
            and total_displacement >= self.escape_max_displacement
        ):
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={total_displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, directional_sidestep stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                total_displacement,
                self.escape_max_displacement,
            )
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        tilt, roll, pitch = self._tilt_abs(robot)
        if (
            self.tilt_recovery_enable
            and self.escape_phase not in {"pre_crouch", "tilt_recover", "finished"}
            and tilt >= self.tilt_warn_rad
        ):
            if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                self._finish_escape("tilt_recover_failed")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                    f"r={roll:.2f} p={pitch:.2f}"
                )
                self._log(
                    "version=%s, directional_sidestep tilt recovery failed before new attempt: "
                    "attempts=%d, roll=%.3f, pitch=%.3f",
                    self.version,
                    self.tilt_recovery_attempts,
                    roll,
                    pitch,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            self._start_tilt_recovery(robot, tilt, roll, pitch)

        if self.escape_phase == "pre_crouch":
            if self.escape_phase_step >= self.crouch_steps:
                self._start_escape_phase("direction_probe_primary", robot)
        elif self.escape_phase == "tilt_recover":
            if (
                self.escape_phase_step >= self.tilt_recovery_min_steps
                and tilt <= self.tilt_clear_rad
                and self.tilt_resume_phase
            ):
                self._start_escape_phase(self.tilt_resume_phase, robot)
            elif self.escape_phase_step >= self.tilt_recovery_steps:
                if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                    self._finish_escape("tilt_recover_failed")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                        f"r={roll:.2f} p={pitch:.2f}"
                    )
                    self._log(
                        "version=%s, directional_sidestep tilt recovery failed: "
                        "attempts=%d, roll=%.3f, pitch=%.3f",
                        self.version,
                        self.tilt_recovery_attempts,
                        roll,
                        pitch,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_tilt_recovery(robot, tilt, roll, pitch)
        elif self.escape_phase == "direction_probe_primary":
            lateral_progress = self._lateral_progress_from_phase(robot, self.direction_primary_side)
            if (
                lateral_progress is not None
                and lateral_progress >= self.lateral_probe_distance
            ) or self.escape_phase_step >= self.lateral_probe_steps:
                self._record_direction_probe_score(self.direction_primary_side, robot, tilt)
                self._start_escape_phase("direction_return_primary", robot)
        elif self.escape_phase == "direction_return_primary":
            remaining_primary_offset = self._lateral_progress_from_start(robot, self.direction_primary_side)
            if (
                remaining_primary_offset is not None
                and remaining_primary_offset <= self.direction_return_tolerance
            ) or self.escape_phase_step >= self.direction_return_steps:
                self._start_escape_phase("direction_probe_secondary", robot)
        elif self.escape_phase == "direction_probe_secondary":
            lateral_progress = self._lateral_progress_from_phase(robot, self.direction_secondary_side)
            if (
                lateral_progress is not None
                and lateral_progress >= self.lateral_probe_distance
            ) or self.escape_phase_step >= self.lateral_probe_steps:
                self._record_direction_probe_score(self.direction_secondary_side, robot, tilt)
                self._choose_direction_side()
                if not self.direction_valid_side:
                    self._finish_escape("direction_no_valid_side")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop no_valid_side "
                        f"min_probe={self.direction_min_probe_progress:.2f}"
                    )
                    self._log(
                        "version=%s, directional_sidestep no valid side after probes: "
                        "scores=%s, lateral_progress=%s, min_probe_progress=%.4f",
                        self.version,
                        self.direction_probe_scores,
                        self.direction_probe_lateral_progress,
                        self.direction_min_probe_progress,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.strafe_cycle = 0
                self.forward_cycle = 1
                self._start_escape_phase("direction_commit_strafe", robot)
        elif self.escape_phase == "direction_commit_strafe":
            phase_lateral = self._lateral_progress_from_phase(robot, self.escape_side)
            net_lateral = self._lateral_progress_from_start(robot, self.escape_side)
            lateral_target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            lateral_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and phase_lateral is not None
                and phase_lateral < self.direction_stall_min_progress
            )
            lateral_reached = net_lateral is not None and net_lateral >= lateral_target
            lateral_timed_out = self.escape_phase_step >= self.direction_commit_lateral_steps
            if lateral_stalled:
                if self.direction_side_switches < self.direction_max_side_switches:
                    old_side = self.escape_side
                    self.direction_side_switches += 1
                    self.escape_side = int(-self.escape_side)
                    self.direction_chosen_side = int(self.escape_side)
                    self._log(
                        "version=%s, directional_sidestep switching lateral side: old_side=%d, "
                        "new_side=%d, phase_lateral=%s, min_progress=%.4f, switches=%d/%d",
                        self.version,
                        old_side,
                        self.escape_side,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        self.direction_stall_min_progress,
                        self.direction_side_switches,
                        self.direction_max_side_switches,
                    )
                    self._start_escape_phase("direction_commit_strafe", robot)
                else:
                    self._finish_escape("direction_lateral_stalled")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop lateral_stalled "
                        f"d={0.0 if phase_lateral is None else phase_lateral:.2f}"
                    )
                    self._log(
                        "version=%s, directional_sidestep lateral stalled: side=%d, "
                        "phase_lateral=%s, min_progress=%.4f, net_lateral=%s, switches=%d",
                        self.version,
                        self.escape_side,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        self.direction_stall_min_progress,
                        None if net_lateral is None else round(net_lateral, 4),
                        self.direction_side_switches,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            elif lateral_reached or lateral_timed_out:
                self.strafe_cycle += 1
                self._start_escape_phase("direction_commit_forward", robot)
        elif self.escape_phase == "direction_commit_forward":
            forward_progress = self._forward_progress_from_phase(robot)
            forward_total = self._forward_progress_from_start(robot)
            forward_reached = (
                (
                    forward_progress is not None
                    and forward_progress >= self.direction_commit_forward_distance
                )
                or (
                    forward_total is not None
                    and forward_total >= self.direction_commit_forward_distance
                )
            )
            forward_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and forward_progress is not None
                and forward_progress < self.direction_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.direction_commit_forward_steps
            if forward_reached:
                self._finish_escape("direction_forward_done")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop direction_forward_done"
                self._log(
                    "version=%s, directional_sidestep done: forward_progress=%s, "
                    "forward_total=%s, target=%.4f, total_displacement=%s, side=%d, strafe_cycles=%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    None if forward_total is None else round(forward_total, 4),
                    self.direction_commit_forward_distance,
                    None if total_displacement is None else round(total_displacement, 4),
                    self.escape_side,
                    self.strafe_cycle,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if forward_stalled or forward_timed_out:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("direction_cycles_exhausted")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop cycles_exhausted "
                        f"fwd={0.0 if forward_progress is None else forward_progress:.2f}"
                    )
                    self._log(
                        "version=%s, directional_sidestep exhausted: forward_progress=%s, "
                        "forward_total=%s, target=%.4f, stalled=%s, timed_out=%s, "
                        "strafe_cycle=%d/%d, total_displacement=%s",
                        self.version,
                        None if forward_progress is None else round(forward_progress, 4),
                        None if forward_total is None else round(forward_total, 4),
                        self.direction_commit_forward_distance,
                        forward_stalled,
                        forward_timed_out,
                        self.strafe_cycle,
                        self.max_strafe_cycles,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.forward_cycle += 1
                self._log(
                    "version=%s, directional_sidestep forward blocked; continuing same-side strafe: "
                    "side=%d, forward_progress=%s, forward_total=%s, target=%.4f, "
                    "stalled=%s, timed_out=%s, strafe_cycle=%d/%d",
                    self.version,
                    self.escape_side,
                    None if forward_progress is None else round(forward_progress, 4),
                    None if forward_total is None else round(forward_total, 4),
                    self.direction_commit_forward_distance,
                    forward_stalled,
                    forward_timed_out,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("direction_commit_strafe", robot)

        self.escape_phase_step += 1
        side_label = "left" if self.escape_side > 0 else "right"
        trunk_target = self._crouch_target_for_action()
        target = 0.0
        progress = 0.0
        mode = self.escape_phase

        if self.escape_phase == "pre_crouch":
            mode = "pre_crouch"
            target = 1.0
            progress = min(1.0, self.escape_phase_step / max(1, self.crouch_steps))
            trunk_target = self._crouch_interpolation_target()
            action = self._escape_action(policy_action, 0.0, 0.0, 0.0, trunk_target, remember_cmd=False)
        elif self.escape_phase == "tilt_recover":
            mode = f"tilt_recover r={roll:.2f} p={pitch:.2f}"
            target = self.tilt_clear_rad
            progress = tilt
            vx, vy, wz = self.tilt_recovery_cmd.tolist()
            action = self._escape_action(policy_action, vx, vy, wz, trunk_target, remember_cmd=False)
        elif self.escape_phase == "direction_probe_primary":
            side = self.direction_primary_side
            side_label = "left" if side > 0 else "right"
            mode = f"probe_primary {side_label}"
            target = self.lateral_probe_distance
            progress = self._lateral_progress_from_phase(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "direction_return_primary":
            side = -self.direction_primary_side
            side_label = "left" if side > 0 else "right"
            mode = f"return_to_center {side_label}"
            target = self.direction_return_tolerance
            remaining = self._lateral_progress_from_start(robot, self.direction_primary_side)
            progress = 0.0 if remaining is None else max(0.0, remaining)
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "direction_probe_secondary":
            side = self.direction_secondary_side
            side_label = "left" if side > 0 else "right"
            mode = f"probe_secondary {side_label}"
            target = self.lateral_probe_distance
            progress = self._lateral_progress_from_phase(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "direction_commit_strafe":
            side = self.escape_side
            side_label = "left" if side > 0 else "right"
            mode = f"commit_strafe {side_label} s{self.strafe_cycle + 1:02d}"
            target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            progress = self._lateral_progress_from_start(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "direction_commit_forward":
            mode = f"commit_forward f{self.forward_cycle:02d}"
            target = self.direction_commit_forward_distance
            progress = self._forward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        else:
            self._finish_escape("unknown_phase")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop unknown_phase"
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        if self.escape_phase == "tilt_recover":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"{self.escape_phase_step}/{self.tilt_recovery_steps} clear={self.tilt_clear_rad:.2f}"
            )
        elif self.escape_phase == "direction_return_primary":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"remain={progress:.2f}/{target:.2f}"
            )
        else:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"d={progress:.2f}/{target:.2f} cyc={self.strafe_cycle}/{self.max_strafe_cycles}"
            )

        if self.escape_phase_step in {1, 30, 60, 90, 120, 180, 240}:
            net_lateral = self._lateral_progress_from_start(robot, self.escape_side)
            forward_progress = self._forward_progress_from_phase(robot)
            self._log(
                "version=%s, directional_sidestep step: phase=%s, side=%d, phase_step=%d, "
                "progress=%.4f, target=%.4f, net_lateral=%s, forward_progress=%s, "
                "total_displacement=%s, roll=%.3f, pitch=%.3f",
                self.version,
                self.escape_phase,
                self.escape_side,
                self.escape_phase_step,
                progress,
                target,
                None if net_lateral is None else round(net_lateral, 4),
                None if forward_progress is None else round(forward_progress, 4),
                None if total_displacement is None else round(total_displacement, 4),
                roll,
                pitch,
            )
        return action

    def _retreat_left_gap_commit_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        total_displacement = self._current_escape_displacement(robot)
        if (
            self.escape_max_displacement > 0
            and total_displacement is not None
            and total_displacement >= self.escape_max_displacement
        ):
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={total_displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, retreat_left stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                total_displacement,
                self.escape_max_displacement,
            )
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        self.escape_side = 1
        tilt, roll, pitch = self._tilt_abs(robot)
        if (
            self.tilt_recovery_enable
            and self.escape_phase not in {"pre_crouch", "tilt_recover", "finished"}
            and tilt >= self.tilt_warn_rad
        ):
            if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                self._finish_escape("tilt_recover_failed")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                    f"r={roll:.2f} p={pitch:.2f}"
                )
                self._log(
                    "version=%s, retreat_left tilt recovery failed before new attempt: "
                    "attempts=%d, roll=%.3f, pitch=%.3f",
                    self.version,
                    self.tilt_recovery_attempts,
                    roll,
                    pitch,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            self._start_tilt_recovery(robot, tilt, roll, pitch)

        if self.escape_phase == "pre_crouch":
            if self.escape_phase_step >= self.crouch_steps:
                self._start_escape_phase("retreat_back", robot)
        elif self.escape_phase == "tilt_recover":
            if (
                self.escape_phase_step >= self.tilt_recovery_min_steps
                and tilt <= self.tilt_clear_rad
                and self.tilt_resume_phase
            ):
                self._start_escape_phase(self.tilt_resume_phase, robot)
            elif self.escape_phase_step >= self.tilt_recovery_steps:
                if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                    self._finish_escape("tilt_recover_failed")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                        f"r={roll:.2f} p={pitch:.2f}"
                    )
                    self._log(
                        "version=%s, retreat_left tilt recovery failed: attempts=%d, roll=%.3f, pitch=%.3f",
                        self.version,
                        self.tilt_recovery_attempts,
                        roll,
                        pitch,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self._start_tilt_recovery(robot, tilt, roll, pitch)
        elif self.escape_phase == "retreat_back":
            back_progress = self._backward_progress_from_phase(robot)
            retreat_reached = back_progress is not None and back_progress >= self.retreat_distance
            retreat_stalled = (
                self.retreat_stall_check_steps > 0
                and self.escape_phase_step >= self.retreat_stall_check_steps
                and back_progress is not None
                and back_progress < self.retreat_stall_min_progress
            )
            retreat_timed_out = self.escape_phase_step >= self.retreat_steps
            if retreat_stalled:
                self._finish_escape("retreat_stalled")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop retreat_stalled "
                    f"d={0.0 if back_progress is None else back_progress:.2f}"
                )
                self._log(
                    "version=%s, retreat_left retreat stalled: back_progress=%s, min_progress=%.4f",
                    self.version,
                    None if back_progress is None else round(back_progress, 4),
                    self.retreat_stall_min_progress,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if retreat_reached or retreat_timed_out:
                self._start_escape_phase("retreat_left_strafe", robot)
        elif self.escape_phase == "retreat_left_strafe":
            phase_lateral = self._lateral_progress_from_phase(robot, 1)
            net_lateral = self._lateral_progress_from_start(robot, 1)
            lateral_target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            lateral_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and phase_lateral is not None
                and phase_lateral < self.direction_stall_min_progress
            )
            lateral_reached = net_lateral is not None and net_lateral >= lateral_target
            lateral_timed_out = self.escape_phase_step >= self.direction_commit_lateral_steps
            if lateral_stalled:
                if self.probe_cycle >= self.max_probe_cycles:
                    self._finish_escape("left_strafe_stalled")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop left_strafe_stalled "
                        f"d={0.0 if phase_lateral is None else phase_lateral:.2f}"
                    )
                    self._log(
                        "version=%s, retreat_left left strafe stalled: phase_lateral=%s, "
                        "net_lateral=%s, retreat_cycle=%d/%d",
                        self.version,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        None if net_lateral is None else round(net_lateral, 4),
                        self.probe_cycle,
                        self.max_probe_cycles,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.probe_cycle += 1
                self._log(
                    "version=%s, retreat_left left strafe blocked; retreating farther: "
                    "phase_lateral=%s, net_lateral=%s, retreat_cycle=%d/%d",
                    self.version,
                    None if phase_lateral is None else round(phase_lateral, 4),
                    None if net_lateral is None else round(net_lateral, 4),
                    self.probe_cycle,
                    self.max_probe_cycles,
                )
                self._start_escape_phase("retreat_back", robot)
            elif lateral_reached or lateral_timed_out:
                self.strafe_cycle += 1
                self._start_escape_phase("retreat_forward_commit", robot)
        elif self.escape_phase == "retreat_forward_commit":
            forward_progress = self._forward_progress_from_phase(robot)
            forward_reached = (
                forward_progress is not None
                and forward_progress >= self.direction_commit_forward_distance
            )
            forward_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and forward_progress is not None
                and forward_progress < self.direction_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.direction_commit_forward_steps
            if forward_reached:
                self._finish_escape("retreat_forward_done")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop retreat_forward_done"
                self._log(
                    "version=%s, retreat_left done: forward_progress=%s, target=%.4f, "
                    "total_displacement=%s, strafe_cycles=%d, retreat_cycles=%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.direction_commit_forward_distance,
                    None if total_displacement is None else round(total_displacement, 4),
                    self.strafe_cycle,
                    self.probe_cycle,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if forward_stalled or forward_timed_out:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("retreat_cycles_exhausted")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop retreat_cycles_exhausted "
                        f"fwd={0.0 if forward_progress is None else forward_progress:.2f}"
                    )
                    self._log(
                        "version=%s, retreat_left exhausted: forward_progress=%s, target=%.4f, "
                        "stalled=%s, timed_out=%s, strafe_cycle=%d/%d, retreat_cycle=%d/%d, "
                        "total_displacement=%s",
                        self.version,
                        None if forward_progress is None else round(forward_progress, 4),
                        self.direction_commit_forward_distance,
                        forward_stalled,
                        forward_timed_out,
                        self.strafe_cycle,
                        self.max_strafe_cycles,
                        self.probe_cycle,
                        self.max_probe_cycles,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.forward_cycle += 1
                self._log(
                    "version=%s, retreat_left forward blocked; continuing left strafe: "
                    "forward_progress=%s, target=%.4f, stalled=%s, timed_out=%s, strafe_cycle=%d/%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.direction_commit_forward_distance,
                    forward_stalled,
                    forward_timed_out,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("retreat_left_strafe", robot)

        self.escape_phase_step += 1
        trunk_target = self._crouch_target_for_action()
        target = 0.0
        progress = 0.0
        mode = self.escape_phase

        if self.escape_phase == "pre_crouch":
            mode = "pre_crouch"
            target = 1.0
            progress = min(1.0, self.escape_phase_step / max(1, self.crouch_steps))
            trunk_target = self._crouch_interpolation_target()
            action = self._escape_action(policy_action, 0.0, 0.0, 0.0, trunk_target, remember_cmd=False)
        elif self.escape_phase == "tilt_recover":
            mode = f"tilt_recover r={roll:.2f} p={pitch:.2f}"
            target = self.tilt_clear_rad
            progress = tilt
            vx, vy, wz = self.tilt_recovery_cmd.tolist()
            action = self._escape_action(policy_action, vx, vy, wz, trunk_target, remember_cmd=False)
        elif self.escape_phase == "retreat_back":
            mode = f"retreat_back r{self.probe_cycle:02d}"
            target = self.retreat_distance
            progress = self._backward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.back_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        elif self.escape_phase == "retreat_left_strafe":
            mode = f"retreat_left_strafe s{self.strafe_cycle + 1:02d}"
            target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            progress = self._lateral_progress_from_start(robot, 1) or 0.0
            action = self._escape_action(policy_action, 0.0, self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "retreat_forward_commit":
            mode = f"retreat_forward f{self.forward_cycle:02d}"
            target = self.direction_commit_forward_distance
            progress = self._forward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        else:
            self._finish_escape("unknown_phase")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop unknown_phase"
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        if self.escape_phase == "tilt_recover":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"{self.escape_phase_step}/{self.tilt_recovery_steps} clear={self.tilt_clear_rad:.2f}"
            )
        else:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"d={progress:.2f}/{target:.2f} strafe={self.strafe_cycle}/{self.max_strafe_cycles}"
            )

        if self.escape_phase_step in {1, 30, 60, 90, 120, 180, 240, 300}:
            back_progress = self._backward_progress_from_phase(robot)
            net_left = self._lateral_progress_from_start(robot, 1)
            forward_progress = self._forward_progress_from_phase(robot)
            self._log(
                "version=%s, retreat_left step: phase=%s, phase_step=%d, progress=%.4f, "
                "target=%.4f, back_progress=%s, net_left=%s, forward_progress=%s, "
                "total_displacement=%s, roll=%.3f, pitch=%.3f",
                self.version,
                self.escape_phase,
                self.escape_phase_step,
                progress,
                target,
                None if back_progress is None else round(back_progress, 4),
                None if net_left is None else round(net_left, 4),
                None if forward_progress is None else round(forward_progress, 4),
                None if total_displacement is None else round(total_displacement, 4),
                roll,
                pitch,
            )
        return action

    def _conditional_retreat_left_gap_step(self, policy_action: Any, robot: Any | None = None) -> torch.Tensor:
        total_displacement = self._current_escape_displacement(robot)
        if (
            self.escape_max_displacement > 0
            and total_displacement is not None
            and total_displacement >= self.escape_max_displacement
        ):
            self._finish_escape("max-displacement")
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} stop d={total_displacement:.2f}>={self.escape_max_displacement:.2f}"
            )
            self._log(
                "version=%s, conditional_retreat stopped by displacement cap: displacement=%.4f cap=%.4f",
                self.version,
                total_displacement,
                self.escape_max_displacement,
            )
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        tilt, roll, pitch = self._tilt_abs(robot)
        if (
            self.tilt_recovery_enable
            and self.escape_phase not in {"pre_crouch", "tilt_recover", "finished"}
            and tilt >= self.tilt_warn_rad
        ):
            if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                routed = self._route_conditional_after_tilt_recovery(
                    robot,
                    tilt,
                    roll,
                    pitch,
                    "max_attempts_before_new_recovery",
                )
                if routed and self.escape_phase == "finished":
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                if not routed:
                    self._finish_escape("tilt_recover_failed")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                        f"r={roll:.2f} p={pitch:.2f}"
                    )
                    self._log(
                        "version=%s, conditional_retreat tilt recovery failed before new attempt: "
                        "attempts=%d, roll=%.3f, pitch=%.3f",
                        self.version,
                        self.tilt_recovery_attempts,
                        roll,
                        pitch,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            else:
                self._start_tilt_recovery(robot, tilt, roll, pitch)

        if self.escape_phase == "pre_crouch":
            if self.escape_phase_step >= self.crouch_steps:
                self._start_escape_phase("conditional_forward_probe", robot)
        elif self.escape_phase == "tilt_recover":
            if (
                self.escape_phase_step >= self.tilt_recovery_min_steps
                and tilt <= self.tilt_clear_rad
                and self.tilt_resume_phase
            ):
                routed = self._route_conditional_after_tilt_recovery(
                    robot,
                    tilt,
                    roll,
                    pitch,
                    "tilt_cleared",
                )
                if routed and self.escape_phase == "finished":
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                if not routed:
                    self._start_escape_phase(self.tilt_resume_phase, robot)
            elif self.escape_phase_step >= self.tilt_recovery_steps:
                if self.tilt_recovery_attempts >= self.tilt_recovery_max_attempts:
                    routed = self._route_conditional_after_tilt_recovery(
                        robot,
                        tilt,
                        roll,
                        pitch,
                        "tilt_recovery_exhausted",
                    )
                    if routed and self.escape_phase == "finished":
                        return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                    if not routed:
                        self._finish_escape("tilt_recover_failed")
                        self.video_label = (
                            f"{self.version} AS#{self.num_triggers} stop tilt_recover_failed "
                            f"r={roll:.2f} p={pitch:.2f}"
                        )
                        self._log(
                            "version=%s, conditional_retreat tilt recovery failed: attempts=%d, roll=%.3f, pitch=%.3f",
                            self.version,
                            self.tilt_recovery_attempts,
                            roll,
                            pitch,
                        )
                        return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                else:
                    self._start_tilt_recovery(robot, tilt, roll, pitch)
        elif self.escape_phase == "conditional_forward_probe":
            forward_progress = self._forward_progress_from_phase(robot)
            forward_reached = forward_progress is not None and forward_progress >= self.forward_probe_distance
            forward_stalled = (
                self.forward_stall_check_steps > 0
                and self.escape_phase_step >= self.forward_stall_check_steps
                and forward_progress is not None
                and forward_progress <= self.forward_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.forward_probe_steps
            if forward_reached:
                self._finish_escape("conditional_forward_done")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop forward_ok "
                    f"fwd={0.0 if forward_progress is None else forward_progress:.2f}"
                )
                self._log(
                    "version=%s, conditional_retreat forward probe succeeded; no retreat needed: "
                    "forward_progress=%s, target=%.4f, total_displacement=%s",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.forward_probe_distance,
                    None if total_displacement is None else round(total_displacement, 4),
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if forward_stalled or forward_timed_out:
                self._log(
                    "version=%s, conditional_retreat forward probe blocked; probing lateral directions: "
                    "forward_progress=%s, target=%.4f, stalled=%s, timed_out=%s, "
                    "phase_step=%d, min_progress=%.4f",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.forward_probe_distance,
                    forward_stalled,
                    forward_timed_out,
                    self.escape_phase_step,
                    self.forward_stall_min_progress,
                )
                self._start_escape_phase("conditional_probe_primary", robot)
        elif self.escape_phase == "conditional_probe_primary":
            lateral_progress = self._lateral_progress_from_phase(robot, self.direction_primary_side)
            if (
                lateral_progress is not None
                and lateral_progress >= self.lateral_probe_distance
            ) or self.escape_phase_step >= self.lateral_probe_steps:
                self._record_direction_probe_score(self.direction_primary_side, robot, tilt)
                self._start_escape_phase("conditional_return_primary", robot)
        elif self.escape_phase == "conditional_return_primary":
            remaining_primary_offset = self._lateral_progress_from_start(robot, self.direction_primary_side)
            if (
                remaining_primary_offset is not None
                and remaining_primary_offset <= self.direction_return_tolerance
            ) or self.escape_phase_step >= self.direction_return_steps:
                self._start_escape_phase("conditional_probe_secondary", robot)
        elif self.escape_phase == "conditional_probe_secondary":
            lateral_progress = self._lateral_progress_from_phase(robot, self.direction_secondary_side)
            if (
                lateral_progress is not None
                and lateral_progress >= self.lateral_probe_distance
            ) or self.escape_phase_step >= self.lateral_probe_steps:
                self._record_direction_probe_score(self.direction_secondary_side, robot, tilt)
                self._choose_direction_side()
                if not self.direction_valid_side:
                    self.escape_side = 1
                    self.probe_cycle = 1
                    self.strafe_cycle = 0
                    self.forward_cycle = 1
                    self._log(
                        "version=%s, conditional_retreat both lateral probes failed; using retreat-left fallback: "
                        "scores=%s, lateral_progress=%s, min_probe_progress=%.4f",
                        self.version,
                        self.direction_probe_scores,
                        self.direction_probe_lateral_progress,
                        self.direction_min_probe_progress,
                    )
                    self._start_escape_phase("retreat_back", robot)
                else:
                    self.strafe_cycle = 0
                    self.forward_cycle = 1
                    self._start_escape_phase("conditional_commit_strafe", robot)
        elif self.escape_phase == "conditional_commit_strafe":
            phase_lateral = self._lateral_progress_from_phase(robot, self.escape_side)
            net_lateral = self._lateral_progress_from_start(robot, self.escape_side)
            lateral_target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            lateral_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and phase_lateral is not None
                and phase_lateral <= self.direction_stall_min_progress
            )
            lateral_reached = net_lateral is not None and net_lateral >= lateral_target
            lateral_timed_out = self.escape_phase_step >= self.direction_commit_lateral_steps
            if lateral_stalled:
                if self.direction_side_switches < self.direction_max_side_switches:
                    old_side = self.escape_side
                    self.direction_side_switches += 1
                    self.escape_side = int(-self.escape_side)
                    self.direction_chosen_side = int(self.escape_side)
                    self._log(
                        "version=%s, conditional_retreat lateral commit stalled; switching side: "
                        "old_side=%d, new_side=%d, phase_lateral=%s, min_progress=%.4f, switches=%d/%d",
                        self.version,
                        old_side,
                        self.escape_side,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        self.direction_stall_min_progress,
                        self.direction_side_switches,
                        self.direction_max_side_switches,
                    )
                    self._start_escape_phase("conditional_commit_strafe", robot)
                else:
                    self.escape_side = 1
                    self.probe_cycle = 1
                    self.strafe_cycle = 0
                    self.forward_cycle = 1
                    self._log(
                        "version=%s, conditional_retreat lateral commit failed; using retreat-left fallback: "
                        "phase_lateral=%s, net_lateral=%s, switches=%d/%d",
                        self.version,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        None if net_lateral is None else round(net_lateral, 4),
                        self.direction_side_switches,
                        self.direction_max_side_switches,
                    )
                    self._start_escape_phase("retreat_back", robot)
            elif lateral_reached or lateral_timed_out:
                self.strafe_cycle += 1
                self._start_escape_phase("conditional_commit_forward", robot)
        elif self.escape_phase == "conditional_commit_forward":
            forward_progress = self._forward_progress_from_phase(robot)
            forward_reached = (
                forward_progress is not None
                and forward_progress >= self.direction_commit_forward_distance
            )
            forward_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and forward_progress is not None
                and forward_progress <= self.direction_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.direction_commit_forward_steps
            if forward_reached:
                self._finish_escape("conditional_forward_commit_done")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop conditional_forward_done"
                self._log(
                    "version=%s, conditional_retreat side path succeeded: forward_progress=%s, "
                    "target=%.4f, total_displacement=%s, side=%d, strafe_cycles=%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.direction_commit_forward_distance,
                    None if total_displacement is None else round(total_displacement, 4),
                    self.escape_side,
                    self.strafe_cycle,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if forward_stalled or forward_timed_out:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self.escape_side = 1
                    self.probe_cycle = 1
                    self.strafe_cycle = 0
                    self.forward_cycle = 1
                    self._log(
                        "version=%s, conditional_retreat side path exhausted; using retreat-left fallback: "
                        "forward_progress=%s, target=%.4f, stalled=%s, timed_out=%s",
                        self.version,
                        None if forward_progress is None else round(forward_progress, 4),
                        self.direction_commit_forward_distance,
                        forward_stalled,
                        forward_timed_out,
                    )
                    self._start_escape_phase("retreat_back", robot)
                else:
                    self.forward_cycle += 1
                    self._log(
                        "version=%s, conditional_retreat forward after sidestep blocked; continuing same-side strafe: "
                        "side=%d, forward_progress=%s, target=%.4f, stalled=%s, timed_out=%s, strafe_cycle=%d/%d",
                        self.version,
                        self.escape_side,
                        None if forward_progress is None else round(forward_progress, 4),
                        self.direction_commit_forward_distance,
                        forward_stalled,
                        forward_timed_out,
                        self.strafe_cycle,
                        self.max_strafe_cycles,
                    )
                    self._start_escape_phase("conditional_commit_strafe", robot)
        elif self.escape_phase == "retreat_back":
            self.escape_side = 1
            back_progress = self._backward_progress_from_phase(robot)
            retreat_reached = back_progress is not None and back_progress >= self.retreat_distance
            retreat_stalled = (
                self.retreat_stall_check_steps > 0
                and self.escape_phase_step >= self.retreat_stall_check_steps
                and back_progress is not None
                and back_progress <= self.retreat_stall_min_progress
            )
            retreat_timed_out = self.escape_phase_step >= self.retreat_steps
            if retreat_stalled:
                self._finish_escape("conditional_retreat_stalled")
                self.video_label = (
                    f"{self.version} AS#{self.num_triggers} stop retreat_stalled "
                    f"d={0.0 if back_progress is None else back_progress:.2f}"
                )
                self._log(
                    "version=%s, conditional_retreat fallback retreat stalled: back_progress=%s, min_progress=%.4f",
                    self.version,
                    None if back_progress is None else round(back_progress, 4),
                    self.retreat_stall_min_progress,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if retreat_reached or retreat_timed_out:
                self._start_escape_phase("retreat_left_strafe", robot)
        elif self.escape_phase == "retreat_left_strafe":
            self.escape_side = 1
            phase_lateral = self._lateral_progress_from_phase(robot, 1)
            net_lateral = self._lateral_progress_from_start(robot, 1)
            lateral_target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            lateral_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and phase_lateral is not None
                and phase_lateral <= self.direction_stall_min_progress
            )
            lateral_reached = net_lateral is not None and net_lateral >= lateral_target
            lateral_timed_out = self.escape_phase_step >= self.direction_commit_lateral_steps
            if lateral_stalled:
                if self.probe_cycle >= self.max_probe_cycles:
                    self._finish_escape("conditional_left_strafe_stalled")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop left_strafe_stalled "
                        f"d={0.0 if phase_lateral is None else phase_lateral:.2f}"
                    )
                    self._log(
                        "version=%s, conditional_retreat fallback left strafe stalled: phase_lateral=%s, "
                        "net_lateral=%s, retreat_cycle=%d/%d",
                        self.version,
                        None if phase_lateral is None else round(phase_lateral, 4),
                        None if net_lateral is None else round(net_lateral, 4),
                        self.probe_cycle,
                        self.max_probe_cycles,
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.probe_cycle += 1
                self._log(
                    "version=%s, conditional_retreat fallback left blocked; retreating farther: "
                    "phase_lateral=%s, net_lateral=%s, retreat_cycle=%d/%d",
                    self.version,
                    None if phase_lateral is None else round(phase_lateral, 4),
                    None if net_lateral is None else round(net_lateral, 4),
                    self.probe_cycle,
                    self.max_probe_cycles,
                )
                self._start_escape_phase("retreat_back", robot)
            elif lateral_reached or lateral_timed_out:
                self.strafe_cycle += 1
                self._start_escape_phase("retreat_forward_commit", robot)
        elif self.escape_phase == "retreat_forward_commit":
            forward_progress = self._forward_progress_from_phase(robot)
            forward_reached = (
                forward_progress is not None
                and forward_progress >= self.direction_commit_forward_distance
            )
            forward_stalled = (
                self.direction_stall_check_steps > 0
                and self.escape_phase_step >= self.direction_stall_check_steps
                and forward_progress is not None
                and forward_progress <= self.direction_stall_min_progress
            )
            forward_timed_out = self.escape_phase_step >= self.direction_commit_forward_steps
            if forward_reached:
                self._finish_escape("conditional_retreat_forward_done")
                self.video_label = f"{self.version} AS#{self.num_triggers} stop retreat_forward_done"
                self._log(
                    "version=%s, conditional_retreat fallback done: forward_progress=%s, target=%.4f, "
                    "total_displacement=%s, strafe_cycles=%d, retreat_cycles=%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.direction_commit_forward_distance,
                    None if total_displacement is None else round(total_displacement, 4),
                    self.strafe_cycle,
                    self.probe_cycle,
                )
                return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
            if forward_stalled or forward_timed_out:
                if self.strafe_cycle >= self.max_strafe_cycles:
                    self._finish_escape("conditional_retreat_cycles_exhausted")
                    self.video_label = (
                        f"{self.version} AS#{self.num_triggers} stop retreat_cycles_exhausted "
                        f"fwd={0.0 if forward_progress is None else forward_progress:.2f}"
                    )
                    self._log(
                        "version=%s, conditional_retreat fallback exhausted: forward_progress=%s, target=%.4f, "
                        "stalled=%s, timed_out=%s, strafe_cycle=%d/%d, retreat_cycle=%d/%d, "
                        "total_displacement=%s",
                        self.version,
                        None if forward_progress is None else round(forward_progress, 4),
                        self.direction_commit_forward_distance,
                        forward_stalled,
                        forward_timed_out,
                        self.strafe_cycle,
                        self.max_strafe_cycles,
                        self.probe_cycle,
                        self.max_probe_cycles,
                        None if total_displacement is None else round(total_displacement, 4),
                    )
                    return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())
                self.forward_cycle += 1
                self._log(
                    "version=%s, conditional_retreat fallback forward blocked; continuing left strafe: "
                    "forward_progress=%s, target=%.4f, stalled=%s, timed_out=%s, strafe_cycle=%d/%d",
                    self.version,
                    None if forward_progress is None else round(forward_progress, 4),
                    self.direction_commit_forward_distance,
                    forward_stalled,
                    forward_timed_out,
                    self.strafe_cycle,
                    self.max_strafe_cycles,
                )
                self._start_escape_phase("retreat_left_strafe", robot)

        self.escape_phase_step += 1
        trunk_target = self._crouch_target_for_action()
        target = 0.0
        progress = 0.0
        mode = self.escape_phase

        if self.escape_phase == "pre_crouch":
            mode = "pre_crouch"
            target = 1.0
            progress = min(1.0, self.escape_phase_step / max(1, self.crouch_steps))
            trunk_target = self._crouch_interpolation_target()
            action = self._escape_action(policy_action, 0.0, 0.0, 0.0, trunk_target, remember_cmd=False)
        elif self.escape_phase == "tilt_recover":
            mode = f"tilt_recover r={roll:.2f} p={pitch:.2f}"
            target = self.tilt_clear_rad
            progress = tilt
            vx, vy, wz = self.tilt_recovery_cmd.tolist()
            action = self._escape_action(policy_action, vx, vy, wz, trunk_target, remember_cmd=False)
        elif self.escape_phase == "conditional_forward_probe":
            mode = "forward_probe"
            target = self.forward_probe_distance
            progress = self._forward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        elif self.escape_phase == "conditional_probe_primary":
            side = self.direction_primary_side
            side_label = "left" if side > 0 else "right"
            mode = f"probe_primary {side_label}"
            target = self.lateral_probe_distance
            progress = self._lateral_progress_from_phase(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "conditional_return_primary":
            side = -self.direction_primary_side
            side_label = "left" if side > 0 else "right"
            mode = f"return_to_center {side_label}"
            target = self.direction_return_tolerance
            remaining = self._lateral_progress_from_start(robot, self.direction_primary_side)
            progress = 0.0 if remaining is None else max(0.0, remaining)
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "conditional_probe_secondary":
            side = self.direction_secondary_side
            side_label = "left" if side > 0 else "right"
            mode = f"probe_secondary {side_label}"
            target = self.lateral_probe_distance
            progress = self._lateral_progress_from_phase(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "conditional_commit_strafe":
            side = self.escape_side
            side_label = "left" if side > 0 else "right"
            mode = f"commit_strafe {side_label} s{self.strafe_cycle + 1:02d}"
            target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            progress = self._lateral_progress_from_start(robot, side) or 0.0
            action = self._escape_action(policy_action, 0.0, side * self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "conditional_commit_forward":
            mode = f"commit_forward f{self.forward_cycle:02d}"
            target = self.direction_commit_forward_distance
            progress = self._forward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        elif self.escape_phase == "retreat_back":
            mode = f"fallback_retreat_back r{self.probe_cycle:02d}"
            target = self.retreat_distance
            progress = self._backward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.back_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        elif self.escape_phase == "retreat_left_strafe":
            mode = f"fallback_left_strafe s{self.strafe_cycle + 1:02d}"
            target = self.direction_commit_lateral_distance * max(1, self.strafe_cycle + 1)
            progress = self._lateral_progress_from_start(robot, 1) or 0.0
            action = self._escape_action(policy_action, 0.0, self.strafe_vel * self.current_escape_scale, 0.0, trunk_target)
        elif self.escape_phase == "retreat_forward_commit":
            mode = f"fallback_forward f{self.forward_cycle:02d}"
            target = self.direction_commit_forward_distance
            progress = self._forward_progress_from_phase(robot) or 0.0
            action = self._escape_action(policy_action, self.forward_vel * self.current_escape_scale, 0.0, 0.0, trunk_target)
        else:
            self._finish_escape("unknown_phase")
            self.video_label = f"{self.version} AS#{self.num_triggers} stop unknown_phase"
            return self._escape_action(policy_action, 0.0, 0.0, 0.0, self._crouch_target_for_action())

        if self.escape_phase == "tilt_recover":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"{self.escape_phase_step}/{self.tilt_recovery_steps} clear={self.tilt_clear_rad:.2f}"
            )
        elif self.escape_phase == "conditional_return_primary":
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"remain={progress:.2f}/{target:.2f}"
            )
        else:
            self.video_label = (
                f"{self.version} AS#{self.num_triggers} {mode} "
                f"d={progress:.2f}/{target:.2f} strafe={self.strafe_cycle}/{self.max_strafe_cycles}"
            )

        if self.escape_phase_step in {1, 30, 60, 90, 120, 180, 240, 300}:
            forward_progress = self._forward_progress_from_phase(robot)
            net_lateral = self._lateral_progress_from_start(robot, self.escape_side)
            back_progress = self._backward_progress_from_phase(robot)
            self._log(
                "version=%s, conditional_retreat step: phase=%s, side=%d, phase_step=%d, "
                "progress=%.4f, target=%.4f, forward_progress=%s, net_lateral=%s, "
                "back_progress=%s, total_displacement=%s, roll=%.3f, pitch=%.3f",
                self.version,
                self.escape_phase,
                self.escape_side,
                self.escape_phase_step,
                progress,
                target,
                None if forward_progress is None else round(forward_progress, 4),
                None if net_lateral is None else round(net_lateral, 4),
                None if back_progress is None else round(back_progress, 4),
                None if total_displacement is None else round(total_displacement, 4),
                roll,
                pitch,
            )
        return action

    def consume_finished_escape(self, robot: Any | None = None) -> bool:
        finished = self.finished_escape
        self.finished_escape = False
        if finished:
            displacement = None
            end_xy = None
            if robot is not None and self.escape_start_xy is not None:
                end_xy = self._robot_xy(robot)
                displacement = float(np.linalg.norm(end_xy - self.escape_start_xy))
            if displacement is not None and displacement < self.escape_min_displacement:
                cooldown_steps = self.ineffective_cooldown_steps
                effect = "ineffective"
            else:
                cooldown_steps = self.cooldown_steps
                effect = "effective" if displacement is not None else "unknown"
            self.cooldown_counter = cooldown_steps
            self._log(
                "version=%s, escape finished; effect=%s, stop_reason=%s, displacement=%s, end_xy=%s, cooldown_steps=%d",
                self.version,
                effect,
                self.escape_stop_reason or "unknown",
                None if displacement is None else round(displacement, 4),
                np.round(end_xy, 4).tolist() if end_xy is not None else None,
                cooldown_steps,
            )
            self.escape_start_xy = None
            self.escape_original_trunk_qpos = None
            self.escape_stop_reason = ""
        return finished
