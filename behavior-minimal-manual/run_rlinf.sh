#!/usr/bin/env bash
set -euo pipefail
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 必填参数
CKPT_PATH="${CKPT_PATH:-CKPT_PATH is required}"
ISAAC_PATH="${ISAAC_PATH:?ISAAC_PATH is required}"
TASK_IDX="${TASK_IDX:?TASK_IDX is required}"
TASK_NAME="${TASK_NAME:?TASK_NAME is required}"
SCENE_MODEL="${SCENE_MODEL:?SCENE_MODEL is required}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:?LOG_ROOT_BASE is required}"
export RLINF_VK_ICD_FILENAMES="${RLINF_VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
# 以上为必填参数

RLINF_ROOT=${RLINF_ROOT:-$(pwd)}
LOG_ROOT="${LOG_ROOT:-${LOG_ROOT_BASE}/rlinf_baseline_pt12_${TASK_IDX}_${TASK_NAME}_${INSTANCE_ID}_${RUN_STAMP}}"
OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$(pwd)/../datasets}
INSTANCE_DIR="${INSTANCE_DIR:-${OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances/scenes/${SCENE_MODEL}/json/${SCENE_MODEL}_task_${TASK_NAME}_instances}"
INSTANCE_ID="${INSTANCE_ID:-0}"
MAX_STEPS="${MAX_STEPS:-512}"
TOTAL_ENVS="${TOTAL_ENVS:-1}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-1}"
GPU_LAYOUT="${GPU_LAYOUT:-2}"
ENV_GPU_LAYOUT="${ENV_GPU_LAYOUT:-3}"
ROLLOUT_GPU_LAYOUT="${ROLLOUT_GPU_LAYOUT:-${GPU_LAYOUT}}"
ACTOR_GPU_LAYOUT="${ACTOR_GPU_LAYOUT:-${ROLLOUT_GPU_LAYOUT}}"
RLINF_CUDA_VISIBLE_DEVICES="${RLINF_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ENV_GPU_LAYOUT},${ROLLOUT_GPU_LAYOUT}}}"
RLINF_SEED="${RLINF_SEED:-}"
RLINF_RAY_ISOLATED="${RLINF_RAY_ISOLATED:-1}"
CLUSTER_NAMESPACE="${CLUSTER_NAMESPACE:-rlinf_eval_pt12_t${TASK_IDX}_${INSTANCE_ID}_${RUN_STAMP}_$$}"
RLINF_RAY_TEMP_DIR="${RLINF_RAY_TEMP_DIR:-/tmp/ri_pt12_t${TASK_IDX}_${INSTANCE_ID}_$$}"

mkdir -p "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${RLINF_CUDA_VISIBLE_DEVICES}"
export RLINF_RAY_ISOLATED CLUSTER_NAMESPACE RLINF_RAY_TEMP_DIR
if [[ "${RLINF_RAY_ISOLATED}" =~ ^(1|true|yes|on)$ && -z "${RLINF_RAY_ADDRESS:-}" ]]; then
  unset RAY_ADDRESS
fi
export RLINF_ENV_GPU_LAYOUT="${ENV_GPU_LAYOUT}"
export RLINF_ROLLOUT_GPU_LAYOUT="${ROLLOUT_GPU_LAYOUT}"
export RLINF_ACTOR_GPU_LAYOUT="${ACTOR_GPU_LAYOUT}"
export RLINF_PRESERVE_CUDA_VISIBLE_DEVICES_GROUPS="${RLINF_PRESERVE_CUDA_VISIBLE_DEVICES_GROUPS:-EnvGroup}"
export RLINF_DEBUG_WORKER_ENV="${RLINF_DEBUG_WORKER_ENV:-1}"
export ISAAC_PATH OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH="${OMNIGIBSON_DATASET_PATH:-${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/}"
export OMNIGIBSON_KEY_PATH="${OMNIGIBSON_KEY_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson.key}"
export OMNIGIBSON_ASSET_PATH="${OMNIGIBSON_ASSET_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/}"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export RLINF_VK_DRIVER_FILES="${RLINF_VK_DRIVER_FILES:-${RLINF_VK_ICD_FILENAMES}}"
export VK_ICD_FILENAMES="${RLINF_VK_ICD_FILENAMES}"
export VK_DRIVER_FILES="${RLINF_VK_DRIVER_FILES}"
export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

{
  printf "started_at=%s\n" "$(date --iso-8601=seconds)"
  printf "rlinf_root=%s\n" "${RLINF_ROOT}"
  git -C "${RLINF_ROOT}" rev-parse --short HEAD | sed "s/^/rlinf_commit=/" || true
  printf "ckpt_path=%s\n" "${CKPT_PATH}"
  printf "model_config=pi05_behavior\n"
  printf "model_family=pi0_5\n"
  printf "log_root=%s\n" "${LOG_ROOT}"
  printf "task=%s\n" "${TASK_NAME}"
  printf "task_idx=%s\n" "${TASK_IDX}"
  printf "scene_model=%s\n" "${SCENE_MODEL}"
  printf "activity_instance_id=%s\n" "${INSTANCE_ID}"
  printf "max_steps=%s\n" "${MAX_STEPS}"
  printf "total_envs=%s\n" "${TOTAL_ENVS}"
  printf "eval_rollout_epoch=%s\n" "${EVAL_ROLLOUT_EPOCH}"
  printf "cuda_visible_devices=%s\n" "${CUDA_VISIBLE_DEVICES}"
  printf "env_gpu_layout=%s\n" "${ENV_GPU_LAYOUT}"
  printf "rollout_gpu_layout=%s\n" "${ROLLOUT_GPU_LAYOUT}"
  printf "actor_gpu_layout=%s\n" "${ACTOR_GPU_LAYOUT}"
  printf "rlinf_seed=%s\n" "${RLINF_SEED:-default}"
  printf "rlinf_ray_isolated=%s\n" "${RLINF_RAY_ISOLATED}"
  printf "cluster_namespace=%s\n" "${CLUSTER_NAMESPACE}"
  printf "rlinf_ray_temp_dir=%s\n" "${RLINF_RAY_TEMP_DIR}"
  printf "instance_dir=%s\n" "${INSTANCE_DIR}"
  printf "skip_intermediate_obs_in_chunk=true\n"
  env | sort | grep '^ANTI_STUCK_' || true
} > "${LOG_ROOT}/run_manifest.txt"

# source "${RLINF_ROOT}/.venv/bin/activate"
export VK_ICD_FILENAMES="${RLINF_VK_ICD_FILENAMES}"
export VK_DRIVER_FILES="${RLINF_VK_DRIVER_FILES}"

EXTRA_OVERRIDES=()
if [[ -n "${RLINF_SEED}" ]]; then
  EXTRA_OVERRIDES+=("actor.seed=${RLINF_SEED}")
  EXTRA_OVERRIDES+=("env.train.seed=${RLINF_SEED}")
  EXTRA_OVERRIDES+=("env.eval.seed=${RLINF_SEED}")
fi
EXTRA_OVERRIDES+=("$@")

python "${EMBODIED_PATH}/eval_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name behavior_ppo_openpi_pi05_eval \
  runner.logger.log_path="${LOG_ROOT}" \
  runner.logger.experiment_name="rlinf_pt12_t${TASK_IDX}_${TASK_NAME}_${INSTANCE_ID}" \
  rollout.model.model_path="${CKPT_PATH}" \
  actor.model.model_path="${CKPT_PATH}" \
  env.train.total_num_envs="${TOTAL_ENVS}" \
  env.eval.total_num_envs="${TOTAL_ENVS}" \
  env.train.max_episode_steps="${MAX_STEPS}" \
  env.eval.max_episode_steps="${MAX_STEPS}" \
  env.train.max_steps_per_rollout_epoch="${MAX_STEPS}" \
  env.eval.max_steps_per_rollout_epoch="${MAX_STEPS}" \
  algorithm.eval_rollout_epoch="${EVAL_ROLLOUT_EPOCH}" \
  env.train.video_cfg.save_video=true \
  env.eval.video_cfg.save_video=true \
  env.train.omni_config.task.activity_name="${TASK_NAME}" \
  env.eval.omni_config.task.activity_name="${TASK_NAME}" \
  env.train.omni_config.scene.scene_model="${SCENE_MODEL}" \
  env.eval.omni_config.scene.scene_model="${SCENE_MODEL}" \
  env.train.omni_config.task.activity_definition_id=0 \
  env.eval.omni_config.task.activity_definition_id=0 \
  env.train.omni_config.task.activity_instance_id="${INSTANCE_ID}" \
  env.eval.omni_config.task.activity_instance_id="${INSTANCE_ID}" \
  env.train.omni_config.task.activity_instance_dir="${INSTANCE_DIR}" \
  env.eval.omni_config.task.activity_instance_dir="${INSTANCE_DIR}" \
  env.train.skip_intermediate_obs_in_chunk=true \
  env.eval.skip_intermediate_obs_in_chunk=true \
  "${EXTRA_OVERRIDES[@]}"
