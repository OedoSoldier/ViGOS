#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a merged model path or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/vigos_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DATASETS="${EVAL_DATASETS:-zli12321/mm-vet,zli12321/mmmu_pro_10options,zli12321/mmmu-pro-vision,zli12321/MMMU,zli12321/MMSI,zli12321/mathverse,zli12321/mathvista,zli12321/realWorldQA}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-vilp-f,vilp-p}"
PASS_K="${PASS_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_TOP_P="${GEN_TOP_P:-0.9}"
GEN_TOP_K="${GEN_TOP_K:-50}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-true}"
GEN_SEED="${GEN_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$BATCH_SIZE}"
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-16}"
VLLM_MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-false}"
DTYPE="${DTYPE:-auto}"
RESUME_RESPONSES="${RESUME_RESPONSES:-false}"
SKIP_JUDGE="${SKIP_JUDGE:-false}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_API_URL="${JUDGE_API_URL:-https://api.deepseek.com}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-512}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16384}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_RETRIES="${JUDGE_RETRIES:-2}"
JUDGE_LOG_EVERY="${JUDGE_LOG_EVERY:-100}"

CMD=(
  uv run python scripts/eval_vigos.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --datasets "$EVAL_DATASETS"
  --benchmarks "$EVAL_BENCHMARKS"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$GEN_TEMPERATURE"
  --top-p "$GEN_TOP_P"
  --top-k "$GEN_TOP_K"
  --seed "$GEN_SEED"
  --pass-k "$PASS_K"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --limit-images "$VLLM_LIMIT_IMAGES"
  --mm-processor-cache-gb "$VLLM_MM_PROCESSOR_CACHE_GB"
  --dtype "$DTYPE"
  --judge-model "$JUDGE_MODEL"
  --judge-api-url "$JUDGE_API_URL"
  --judge-key-env "$JUDGE_KEY_ENV"
  --judge-workers "$JUDGE_WORKERS"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --judge-timeout "$JUDGE_TIMEOUT"
  --judge-retries "$JUDGE_RETRIES"
  --judge-log-every "$JUDGE_LOG_EVERY"
)

if [[ "$GEN_DO_SAMPLE" == "true" ]]; then
  CMD+=(--do-sample)
else
  CMD+=(--no-do-sample)
fi
if [[ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" == "true" ]]; then
  CMD+=(--disable-custom-all-reduce)
else
  CMD+=(--no-disable-custom-all-reduce)
fi
if [[ "$RESUME_RESPONSES" == "true" ]]; then
  CMD+=(--resume-responses)
else
  CMD+=(--no-resume-responses)
fi
if [[ "$SKIP_JUDGE" == "true" ]]; then
  CMD+=(--skip-judge)
fi

"${CMD[@]}"
