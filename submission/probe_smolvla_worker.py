#!/usr/bin/env python3
"""Exercise real adapter -> NPZ -> worker -> checkpoint inference with RGBA bytes.

This is an input-contract check with synthetic camera pixels, not a simulator
rollout or a task-success measurement. Run in the supported SmolVLA environment.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.smolvla.deploy_policy import SmolVLAWorkerClient, encode_obs
from submission.test_smolvla_images import camera_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="One SmolVLA task or all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    manifest = json.loads((ROOT / "submission/selected_checkpoints.json").read_text())
    rows = [r for r in manifest["tasks"] if r["policy"] == "smolvla"
            and args.task in ("all", r["task"])]
    if not rows:
        parser.error("Select a SmolVLA task or all")
    results = []
    for row in rows:
        started = time.perf_counter()
        client = SmolVLAWorkerClient({
            "checkpoint_dir": str(ROOT / row["local_checkpoint_dir"]),
            "smolvla_python": sys.executable, "pytorch_device": args.device,
            "smolvla_steps": 10,
        })
        try:
            encoded = encode_obs(camera_observation(), client.image_keys)
            assert all(value.dtype == np.uint8 for key, value in encoded.items()
                       if key.startswith("observation.images."))
            # Reset before each call forces fresh inference rather than reuse of
            # a previous episode's queued actions.
            for _ in range(2):
                client.reset()
                actions = np.asarray(client.infer(encoded, row["task"].replace("_", " ")))
                if actions.shape != (10, 14) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid action chunk: {actions.shape}")
            client.reset()
            result = {"task": row["task"], "hf_revision": row["hf_revision"],
                      "input": "three batched 480x640 RGBA uint8 cameras via encode_obs",
                      "action_shape_per_call": [10, 14], "fresh_inference_calls": 2,
                      "reset_calls": 3, "finite_actions": True,
                      "seconds": round(time.perf_counter() - started, 3)}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            client.close()
    distribution = importlib.metadata.distribution("lerobot")
    evidence = {
        "evaluation": "eval_412bc052c68847", "device": args.device,
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
        "transformers": importlib.metadata.version("transformers"),
        "lerobot_source": json.loads(distribution.read_text("direct_url.json")),
        "test_scope": "Real worker RPC and checkpoint inference on synthetic byte cameras; no simulator or success rate.",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
