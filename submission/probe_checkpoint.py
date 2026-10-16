#!/usr/bin/env python3
"""Load one released checkpoint and check a synthetic action on CPU or CUDA.

This checks model/config/processor compatibility. It is not a simulation rollout
and does not measure task success or the official inference metric.
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "submission/selected_checkpoints.json").read_text())
    rows = {row["task"]: row for row in manifest["tasks"]}
    if args.task not in rows:
        parser.error("Unknown task.")
    row = rows[args.task]
    checkpoint = ROOT / row["local_checkpoint_dir"]
    if not (checkpoint / "model.safetensors").is_file():
        parser.error("Download the pinned checkpoint first.")

    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.perf_counter()
    if row["policy"] in ("act", "dp"):
        adapter = importlib.import_module(f'policy.{row["policy"]}.deploy_policy')
        model = adapter.get_model({"checkpoint_path": str(checkpoint),
                                   "device": args.device, "dp_num_inference_steps": 10})
    else:
        from policy.smolvla.smolvla_worker import load_pinned_policy
        from lerobot.processor import (
            PolicyProcessorPipeline, batch_to_transition, policy_action_to_transition,
            transition_to_batch, transition_to_policy_action,
        )

        model = load_pinned_policy(str(checkpoint), args.device)
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint), config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": args.device}},
            to_transition=batch_to_transition, to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(checkpoint), config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
        )
    # Verify that a permissive upstream loader did not leave unmatched weights.
    # load_model handles shared/tied tensors in the safetensors serialization.
    from safetensors.torch import load_model

    load_model(model, str(checkpoint / "model.safetensors"), strict=True, device=args.device)
    model.to(args.device)
    model.eval()
    load_seconds = time.perf_counter() - started
    batch = {key: torch.zeros((1, *feature.shape), device=args.device)
             for key, feature in model.config.input_features.items()}
    if row["policy"] == "smolvla":
        # Exercise the raw byte contract too; the original all-float fixture
        # could not detect the evaluation's uint8 resize failure.
        import numpy as np
        from policy.smolvla.smolvla_worker import prepare_observation

        raw = {}
        for key, feature in model.config.input_features.items():
            if key.startswith("observation.images."):
                raw[key] = (np.arange(np.prod(feature.shape)) % 256).astype(np.uint8).reshape(feature.shape)
            else:
                raw[key] = np.zeros(feature.shape, dtype=np.float32)
        batch = prepare_observation(raw, args.task.replace("_", " "))
        batch = preprocessor(batch)
    model.reset()
    started = time.perf_counter()
    with torch.inference_mode():
        action = model.select_action(batch)
        if row["policy"] == "smolvla":
            action = postprocessor(action)
    action_seconds = time.perf_counter() - started
    if tuple(action.shape) != (1, 14) or not torch.isfinite(action).all().item():
        raise ValueError(f"Expected finite [1, 14] action, got {tuple(action.shape)}.")
    model.reset()
    result = {"task": args.task, "policy": row["policy"], "hf_revision": row["hf_revision"],
              "device": args.device, "model_load_seconds": load_seconds,
              "synthetic_action_seconds": action_seconds, "action_shape": list(action.shape),
              "finite_actions": True, "strict_weight_match": True, "reset_called": True,
              "test_scope": "Synthetic model inference only; no simulation or success rate."}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
