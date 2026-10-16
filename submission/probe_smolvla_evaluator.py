#!/usr/bin/env python3
"""Run the repository evaluator loop with real SmolVLA and a controlled Gym fixture.

Only simulator construction, expert feasibility and video encoding are replaced.
The evaluator functions are compiled unchanged from scripts/eval_policy.py to
avoid importing the unavailable EmbodiChain runtime on this CPU host. Its actual
argument parser, adapter loader, reset/episode loop, timing aggregation and JSON
writer execute. This is NOT an official GPU evaluation or a task-success test.
"""

import argparse
import ast
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.smolvla import deploy_policy as adapter
import policy.smolvla as policy_package
from submission.test_smolvla_images import camera_observation


class ControlledEnv:
    """Episode 1 times out at 61 steps; episode 2 signals success at step 2."""
    def __init__(self):
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.single_action_space = gym.spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        self.action_space = self.single_action_space
        self.obs = camera_observation()
        self.obs["robot"] = {key: torch.from_numpy(value) for key, value in self.obs["robot"].items()}
        for sensor in self.obs["sensor"].values():
            sensor["color"] = torch.from_numpy(sensor["color"])
        self.episode = -1
        self.steps = 0
        self.closed = False
        self.reset_seeds = []
        self.total_steps = 0

    def reset(self, *, seed, options):
        assert options == {"save_data": False}
        self.episode += 1
        self.steps = 0
        self.reset_seeds.append(seed)
        return self.obs, {"elapsed_steps": torch.tensor([0])}

    def get_wrapper_attr(self, name):
        if name == "is_task_success":
            return lambda: self.episode == 1 and self.steps >= 2
        raise AttributeError(name)

    def step(self, action):
        if self.steps >= (61 if self.episode == 0 else 2):
            raise AssertionError("Adapter stepped past the fixture's episode boundary")
        assert action.shape == (1, 14) and action.dtype == torch.float32
        assert action.device.type == "cpu" and torch.isfinite(action).all()
        self.steps += 1
        self.total_steps += 1
        terminated = self.episode == 1 and self.steps == 2
        truncated = self.episode == 0 and self.steps == 61
        return self.obs, 0.0, torch.tensor([terminated]), torch.tensor([truncated]), {
            "elapsed_steps": torch.tensor([self.steps])}

    def close(self):
        self.closed = True


class RecordingSpy:
    def __init__(self, directory):
        self.save_dir = directory
        self.starts = []
        self.frames = 0
        self.closes = []

    def start_episode(self, episode, seed):
        self.starts.append((episode, seed))

    def record(self, obs):
        self.frames += 1

    def close_episode(self, success=False):
        self.closes.append(success)


def evaluator_functions():
    path = ROOT / "scripts/eval_policy.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = ast.Module(body=[node for node in tree.body if isinstance(
        node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
    namespace = dict(globals(), REPO_ROOT=str(ROOT), SCRIPT_DIR=str(path.parent),
                     POLICY_DIR=str(ROOT / "policy"))
    exec(compile(functions, str(path), "exec"), namespace)
    return namespace


def probe(row, device, directory):
    env = ControlledEnv()
    recorder = RecordingSpy(directory)
    evaluator = evaluator_functions()
    original_find_gym = evaluator["find_gym_config"]
    def find_gym(config):
        gym_config = original_find_gym(config)
        gym_config["max_episode_steps"] = 61  # Test fixture only; never saved to source.
        return gym_config
    evaluator.update({
        "find_gym_config": find_gym,
        "make_env_from_configs": lambda config, gym_config, action_config: (env, gym_config),
        "create_eval_run_dir": lambda config: directory,
        "create_video_recorder": lambda config, run_dir: recorder,
    })
    models, calls = [], []
    original_get_model, original_eval = policy_package.get_model, policy_package.eval
    def get_model(config):
        model = original_get_model(config)
        models.append(model)
        return model
    def evaluate(*args):
        result = original_eval(*args)
        # These are the actual values returned to the unchanged evaluator loop.
        assert len(result) == 4
        calls.append({"elapsed_steps": result[1]["elapsed_steps"].item(),
                      "new_times": list(result[3]),
                      "worker_times": list(args[1].last_inference_times_s)})
        return result
    argv = [str(ROOT / "scripts/eval_policy.py"), "--config", str(ROOT / "policy/smolvla/deploy_policy.yml"),
            "--overrides", "--task_name", row["task"], "--setting", "random",
            "--checkpoint_dir", str(ROOT / row["local_checkpoint_dir"]),
            "--pytorch_device", device, "--smolvla_python", sys.executable,
            "--max_episodes", "2", "--seed", "17", "--eval_expert_check", "False",
            "--headless", "True"]
    log = io.StringIO()
    try:
        with patch.object(sys, "argv", argv), patch.object(policy_package, "get_model", side_effect=get_model), \
                patch.object(policy_package, "eval", side_effect=evaluate), redirect_stdout(log), redirect_stderr(log):
            evaluator["main"]()
    finally:
        for model in models:
            adapter.close_model(model)
        (directory / "synthetic_evaluator.log").write_text(log.getvalue())
    metrics = json.loads((directory / "evaluation_metrics.json").read_text())
    summary = metrics["summary"]
    if summary["episode_count"] != 2 or summary["inference_call_count"] != 3:
        raise AssertionError(f"Unexpected fixture counts: {summary}")
    assert [episode["action_steps"] for episode in metrics["episodes"]] == [61, 2]
    pattern = [len(call["new_times"]) for call in calls]
    assert pattern == [1, 0, 0, 0, 0, 1, 0, 1], f"Unexpected timing pattern: {pattern}"
    assert all(len(call["new_times"]) == len(call["worker_times"]) for call in calls)
    for call in calls:
        for elapsed in call["new_times"] + call["worker_times"]:
            assert np.isfinite(elapsed) and elapsed > 0
    assert env.closed and len(env.reset_seeds) == 2 and len(recorder.starts) == 2
    assert recorder.frames > 0 and recorder.closes == [False, True, False]
    return {
        "task": row["task"], "hf_revision": row["hf_revision"],
        "fixture_episodes_completed": 2, "fixture_env_steps": env.total_steps,
        "four_value_adapter_calls": len(calls), "actual_model_refills": 3,
        "cached_only_adapter_calls": 5, "calls": calls,
        "reset_video_cleanup_and_metrics_json": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="One SmolVLA task or all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    rows = json.loads((ROOT / "submission/selected_checkpoints.json").read_text())["tasks"]
    rows = [row for row in rows if row["policy"] == "smolvla" and args.task in ("all", row["task"])]
    if not rows:
        parser.error("Select a SmolVLA task or all")
    results = []
    for row in rows:
        directory = ROOT / "submission/local_checks/correction2" / row["task"]
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = probe(row, args.device, directory)
        result["seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        print(json.dumps(result), flush=True)
    distribution = importlib.metadata.distribution("lerobot")
    evidence = {
        "evaluation": "eval_412bc052c68847", "device": args.device,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "lerobot_source": json.loads(distribution.read_text("direct_url.json")),
        "test_scope": "Unchanged local evaluator functions and real checkpoints/workers with a controlled Gym fixture. No physics, official run_evaluation.py, held-out seeds or task-success evaluation.",
        "results": results,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
