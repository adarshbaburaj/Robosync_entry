#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy.inference_timing import finish_inference, start_inference


def load_pinned_policy(checkpoint_dir: str, device: str):
    """Robosync_entry: load the complete policy with an immutable VLM dependency."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(
        checkpoint_dir, cli_overrides=[f"--device={device}"]
    )
    if config.vlm_model_name == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        config.vlm_model_name = snapshot_download(
            repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            ignore_patterns=["onnx/*"],
            token=False,
        )
    return SmolVLAPolicy.from_pretrained(checkpoint_dir, config=config, strict=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    # Robosync_entry: honor the requested runtime device when loading and reject
    # incomplete/incompatible checkpoint weights instead of permissive loading.
    policy = load_pinned_policy(checkpoint_dir, args.device)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def prepare_observation(obs: dict[str, np.ndarray], task: str) -> dict:
    """Convert adapter RGB bytes to LeRobot's float32 [0, 1] image contract.

    The frozen checkpoints use VISUAL=IDENTITY, so their processor does not
    perform this conversion. Leave state/qvel/qf values in their original units.
    Already-normalized floating-point images must not be divided a second time.
    """
    tensors = {}
    for key, value in obs.items():
        tensor = torch.from_numpy(value)
        if key.startswith("observation.images."):
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            elif tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
                if not torch.isfinite(tensor).all() or not ((tensor >= 0) & (tensor <= 1)).all():
                    raise ValueError(f"{key}: floating-point images must be finite and in [0, 1]")
            else:
                raise TypeError(f"{key}: expected uint8 [0, 255] or floating-point [0, 1]")
        tensors[key] = tensor
    tensors["task"] = task
    return tensors


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                num_actions = int(request.get("n_action_steps", 1))
                if not 1 <= num_actions <= policy.config.n_action_steps:
                    raise ValueError("n_action_steps must be between 1 and the checkpoint's n_action_steps")
                # At most one real queue refill can occur within this request.
                # Synchronize the worker's own CUDA device, whose ordinal may
                # differ from the simulator parent's CUDA_VISIBLE_DEVICES.
                started_at = start_inference(args.device)
                obs, task = _load_obs(request["obs_path"])
                obs = prepare_observation(obs, task)
                proc_obs = preprocessor(obs)

                actions = []
                inference_count = 0
                for _ in range(num_actions):
                    inference_count += int(policy._check_get_actions_condition())
                    action = policy.select_action(proc_obs)
                    action = postprocessor(action)
                    actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())
                inference_times_s = []
                if inference_count:
                    finish_inference(started_at, inference_times_s, args.device)

            _send({"ok": True, "actions": actions,
                   "inference_times_s": inference_times_s})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
