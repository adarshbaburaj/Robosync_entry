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
uv pip install --python "$worker_dir/bin/python" \
    'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@b9cb121cb7d4e3e26ec5c906d08dda68d151ba52'

"$worker_dir/bin/python" -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, policy_action_to_transition, transition_to_batch, transition_to_policy_action; print("SmolVLA worker imports passed. Checkpoint loading and GPU inference still need verification.")'
printf 'Set SMOLVLA_PYTHON to %s/bin/python\n' "$worker_dir"
