#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
worker_dir="$repo_root/policy/smolvla/.venv-worker"

if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Follow the official runtime installation guide.' >&2
    exit 1
fi

if [[ ! -x "$worker_dir/bin/python" ]]; then
    uv venv --python 3.12 "$worker_dir"
fi
torch_backend=auto
if [[ "$(uname -s)" == Linux ]]; then
    torch_backend=cu128
fi
uv pip install --python "$worker_dir/bin/python" --torch-backend "$torch_backend" \
    'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@146d8474932c8b566717eb2738ba459337ab71c4' \
    'torch==2.7.1' 'torchvision==0.22.1' 'transformers==5.5.4' \
    --constraint "$script_dir/smolvla_runtime_constraints.txt"

"$worker_dir/bin/python" -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, policy_action_to_transition, transition_to_batch, transition_to_policy_action; print("SmolVLA worker imports passed. Checkpoint loading and GPU inference still need verification.")'
printf 'Set SMOLVLA_PYTHON to %s/bin/python\n' "$worker_dir"
