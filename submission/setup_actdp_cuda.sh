#!/usr/bin/env bash
# After the official frozen ACT/DP environment installation, select the CUDA
# 12.8 builds used by the published RTX 5090 protocol.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This CUDA helper requires a Linux NVIDIA environment.' >&2
    exit 1
fi
for policy in act dp; do
    policy_python="$repo_root/policy/$policy/.venv/bin/python"
    if [[ ! -x "$policy_python" ]]; then
        echo "First run: uv sync --frozen --project policy/$policy" >&2
        exit 1
    fi
    uv pip install --python "$policy_python" \
        --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
    "$policy_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; x = torch.ones(1, device="cuda"); assert (x * 2).item() == 2; print(torch.__version__, torch.cuda.get_device_name(0))'
done
