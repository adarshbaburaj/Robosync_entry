"""Evaluator-facing adapter regression checks; no robot simulator required."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy import inference_timing
from policy.smolvla import deploy_policy as adapter
from submission.test_smolvla_images import camera_observation


class ContractEnv:
    """A strict Gym-shaped fixture, not a physics or task-success simulation."""
    def __init__(self, *, success_at=None, truncate_at=None, terminate_at=None):
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.single_action_space = SimpleNamespace(
            shape=(14,), low=np.full(14, -1, np.float32), high=np.full(14, 1, np.float32))
        self.steps = 0
        self.success_at = success_at
        self.truncate_at = truncate_at
        self.terminate_at = terminate_at
        self.obs = camera_observation()
        self._current_instruction = "open drawer and place item"

    def get_wrapper_attr(self, name):
        if name != "is_task_success":
            raise AttributeError(name)
        return lambda: self.success_at is not None and self.steps >= self.success_at

    def step(self, action):
        self.steps += 1
        assert action.shape == (1, 14) and action.dtype == torch.float32
        assert torch.isfinite(action).all()
        terminated = self.terminate_at is not None and self.steps >= self.terminate_at
        truncated = self.truncate_at is not None and self.steps >= self.truncate_at
        return self.obs, 0.0, torch.tensor([terminated]), torch.tensor([truncated]), {
            "elapsed_steps": torch.tensor([self.steps])}


class ContractModel:
    image_keys = None
    rescale_gripper = True
    gripper_indices = (6, 13)
    gripper_source_range = (0, 1)

    def __init__(self, fresh=True):
        self.last_inference_times_s = [0.1] if fresh else []
        self.task = None

    def infer(self, obs, task):
        self.task = task
        return [np.full(14, 0.5, np.float32) for _ in range(10)]

    def maybe_print_action_debug(self, actions):
        pass

    def maybe_print_env_action_debug(self, action):
        pass


class AdapterContractTest(unittest.TestCase):
    def test_four_values_real_timing_and_env_step_excluded(self):
        env, model = ContractEnv(), ContractModel()
        # A deterministic clock establishes the exact timing boundary without
        # depending on CPU load. env.step must occur after finish_inference.
        clock = iter([100.0, 100.75])
        with patch.object(inference_timing.time, "perf_counter", side_effect=lambda: next(clock)):
            original_step = env.step
            def step(action):
                self.assertEqual(model.last_inference_times_s, [0.1])
                return original_step(action)
            with patch.object(env, "step", side_effect=step), patch.object(
                adapter, "finish_inference", wraps=adapter.finish_inference
            ) as finish:
                def checked_step(action):
                    self.assertEqual(finish.call_count, 1)
                    return step(action)
                env.step.side_effect = checked_step
                obs, info, truncated, new_times = adapter.eval(env, model, env.obs)
        self.assertEqual(new_times, [0.75])
        self.assertEqual(info["elapsed_steps"].item(), 10)
        self.assertIs(obs, env.obs)
        self.assertFalse(truncated.any())
        self.assertEqual(model.task, env._current_instruction)

    def test_cached_actions_return_empty_timing(self):
        env, model = ContractEnv(), ContractModel(fresh=False)
        with patch.object(adapter, "finish_inference") as finish:
            result = adapter.eval(env, model, env.obs)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[3], [])
        finish.assert_not_called()

    def test_stops_at_success_timeout_and_termination(self):
        for flags in ({"success_at": 2}, {"truncate_at": 3}, {"terminate_at": 1}):
            with self.subTest(flags=flags):
                env = ContractEnv(**flags)
                result = adapter.eval(env, ContractModel(), env.obs)
                self.assertEqual(len(result), 4)
                self.assertEqual(env.steps, next(iter(flags.values())))

    def test_timing_helper_synchronizes_requested_cuda_device(self):
        samples = []
        with patch.object(inference_timing.torch.cuda, "synchronize") as sync, patch.object(
            inference_timing.time, "perf_counter", side_effect=[10.0, 10.25]
        ):
            started = inference_timing.start_inference("cuda:2")
            inference_timing.finish_inference(started, samples, "cuda:2")
        self.assertEqual(samples, [0.25])
        self.assertEqual(sync.call_args_list, [unittest.mock.call("cuda:2")] * 2)

    def test_worker_actions_and_timing_protocol_are_validated(self):
        client = object.__new__(adapter.SmolVLAWorkerClient)
        client.smolvla_steps, client.action_dim = 10, 14
        valid = {"actions": np.zeros((10, 14)).tolist(), "inference_times_s": [0.2]}
        cases = [valid, {**valid, "inference_times_s": []},
                 {**valid, "actions": []}, {**valid, "actions": [[float("nan")] * 14] * 10},
                 {**valid, "inference_times_s": [0]}, {**valid, "inference_times_s": [float("nan")]},
                 {**valid, "inference_times_s": [0.1, 0.2]}]
        for i, response in enumerate(cases):
            with self.subTest(i=i), patch.object(client, "_rpc", return_value=response):
                if i < 2:
                    self.assertEqual(len(client.infer({}, "task")), 10)
                    self.assertEqual(client.last_inference_times_s, response["inference_times_s"])
                else:
                    with self.assertRaises(ValueError):
                        client.infer({}, "task")
                    self.assertEqual(client.last_inference_times_s, [])

    def test_step_count_rejected_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({"n_action_steps": 50}))
            for steps in (0, -1, 51):
                with self.subTest(steps=steps), patch.object(adapter.subprocess, "Popen") as launch:
                    with self.assertRaises(ValueError):
                        adapter.get_model({"checkpoint_dir": directory, "smolvla_steps": steps})
                    launch.assert_not_called()

    def test_official_launcher_routes_task_checkpoint_and_worker(self):
        with tempfile.TemporaryDirectory(prefix="smolvla launch ") as directory:
            capture = Path(directory, "capture.json")
            executable = Path(directory, "python capture")
            executable.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'json.dump({"args":sys.argv[1:],"cuda_mask":os.getenv("CUDA_VISIBLE_DEVICES")},'
                'open(os.environ["SMOLVLA_TEST_CAPTURE"],"w"))\n')
            executable.chmod(0o755)
            rows = json.loads((ROOT / "submission/selected_checkpoints.json").read_text())["tasks"]
            for row in (r for r in rows if r["policy"] == "smolvla"):
                with self.subTest(task=row["task"]):
                    env = {**os.environ, "PYTHON_BIN": str(executable),
                           "SMOLVLA_PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": "9",
                           "SMOLVLA_TEST_CAPTURE": str(capture)}
                    subprocess.run(["bash", str(ROOT / "policy/smolvla/eval.sh"), row["task"],
                                    "random", str(ROOT / row["local_checkpoint_dir"]), "2",
                                    "--headless", "True"], env=env, check=True,
                                   capture_output=True, text=True)
                    recorded = json.loads(capture.read_text())
                    args = recorded["args"]
                    self.assertEqual((ROOT / args[0]).resolve(), ROOT / "scripts/eval_policy.py")
                    overrides = dict(zip(args[4::2], args[5::2]))
                    self.assertEqual(overrides["--task_name"], row["task"])
                    self.assertEqual(overrides["--checkpoint_dir"], str(ROOT / row["local_checkpoint_dir"]))
                    self.assertEqual(overrides["--smolvla_python"], sys.executable)
                    self.assertEqual(overrides["--smolvla_cuda_visible_devices"], "2")
                    self.assertIsNone(recorded["cuda_mask"])


if __name__ == "__main__":
    unittest.main()
