#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATASET_NAME:?Set DATASET_NAME to the HuggingFace training dataset id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-ViGOS}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero2_lora_gpu_8.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/vigos_qwen25_3b_${RUN_ID}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-false}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-3}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.1}"
GENERATION_TOP_P="${GENERATION_TOP_P:-0.95}"
GENERATION_TOP_K="${GENERATION_TOP_K:-20}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
LAMBDA_PERCEPTION="${LAMBDA_PERCEPTION:-1.0}"
LAMBDA_REASONING="${LAMBDA_REASONING:-1.0}"
LAMBDA_REF="${LAMBDA_REF:-2.0}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.05}"
DESCRIPTION_LAST_TOKEN_CLIP="${DESCRIPTION_LAST_TOKEN_CLIP:-0.05}"
REASONING_FIRST_TOKEN_CLIP="${REASONING_FIRST_TOKEN_CLIP:-0.05}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MIN_P="${MIN_P:-0.0}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-false}"
VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-600}"
VLLM_SERVER_GROUP_PORT="${VLLM_SERVER_GROUP_PORT:-51216}"
VLLM_SERVER_REQUEST_BATCH_SIZE="${VLLM_SERVER_REQUEST_BATCH_SIZE:-}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-2}"
COMPLETION_LOG_MAX_SAMPLES="${COMPLETION_LOG_MAX_SAMPLES:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
REPORT_TO="${REPORT_TO:-wandb}"

GRADIENT_CHECKPOINTING_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  GRADIENT_CHECKPOINTING_ARGS=(--gradient_checkpointing)
fi

LIMIT_ARGS=()
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  LIMIT_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_STEPS" ]]; then
  LIMIT_ARGS+=(--max_steps "$MAX_STEPS")
fi

VLLM_SERVER_ARGS=()
if [[ -n "$VLLM_SERVER_BASE_URL" ]]; then
  VLLM_SERVER_ARGS+=(--vllm_server_base_url "$VLLM_SERVER_BASE_URL")
fi
if [[ -n "$VLLM_SERVER_REQUEST_BATCH_SIZE" ]]; then
  VLLM_SERVER_ARGS+=(--vllm_server_request_batch_size "$VLLM_SERVER_REQUEST_BATCH_SIZE")
fi

RUN_CONFIG="${RUN_CONFIG:-vigos_qwen25_3b_gen${MAX_COMPLETION_LENGTH}_mb${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_np${NUM_PROCESSES}_vllm_${VLLM_MODE}}"

uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --main_process_port "${MAIN_PROCESS_PORT:-13378}" \
  vigos/train_vigos.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT" \
  --filter_tiny_images "$FILTER_TINY_IMAGES" \
  --min_image_size "$MIN_IMAGE_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "vigos_qwen25_3b_${RUN_ID}" \
  --run_config "$RUN_CONFIG" \
  "${LIMIT_ARGS[@]}" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --max_grad_norm 0.1 \
  --bf16 \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --generation_temperature "$GENERATION_TEMPERATURE" \
  --generation_top_p "$GENERATION_TOP_P" \
  --generation_top_k "$GENERATION_TOP_K" \
  --distill_temperature "$DISTILL_TEMPERATURE" \
  --lambda_perception "$LAMBDA_PERCEPTION" \
  --lambda_reasoning "$LAMBDA_REASONING" \
  --lambda_ref "$LAMBDA_REF" \
  --token_loss_clip "$TOKEN_LOSS_CLIP" \
  --description_last_token_clip "$DESCRIPTION_LAST_TOKEN_CLIP" \
  --reasoning_first_token_clip "$REASONING_FIRST_TOKEN_CLIP" \
  --presence_penalty "$PRESENCE_PENALTY" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --min_p "$MIN_P" \
  --use_vllm "$USE_VLLM" \
  --vllm_mode "$VLLM_MODE" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_sync_frequency "$VLLM_SYNC_FREQUENCY" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_disable_custom_all_reduce "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" \
  --vllm_server_host "$VLLM_SERVER_HOST" \
  --vllm_server_port "$VLLM_SERVER_PORT" \
  --vllm_server_timeout "$VLLM_SERVER_TIMEOUT" \
  --vllm_server_group_port "$VLLM_SERVER_GROUP_PORT" \
  "${VLLM_SERVER_ARGS[@]}" \
  --completion_log_steps "$COMPLETION_LOG_STEPS" \
  --completion_log_max_samples "$COMPLETION_LOG_MAX_SAMPLES" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --logging_steps "$LOGGING_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
  --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS" \
  --report_to "$REPORT_TO" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES"
