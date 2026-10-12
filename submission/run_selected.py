#!/usr/bin/env python3
"""Download pinned releases and invoke the existing official policy wrappers."""

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "download", "run"))
    parser.add_argument("task", nargs="?", default="all")
    parser.add_argument("--setting", choices=("random", "clear"), default="random")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    manifest = json.loads((ROOT / "submission" / "selected_checkpoints.json").read_text())
    tasks = {row["task"]: row for row in manifest["tasks"]}
    if args.task != "all" and args.task not in tasks:
        parser.error("Unknown task. Use the list action to see all ten tasks.")
    if args.action == "run" and args.task == "all":
        parser.error("Run one task at a time with its official policy environment.")
    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if extra and args.action != "run":
        parser.error("Extra policy options are accepted only for run.")
    if len(extra) % 2:
        parser.error("Extra policy overrides must be --key value pairs.")
    for key in extra[::2]:
        if key in ("--task_name", "--policy_name", "--checkpoint_path", "--checkpoint_dir"):
            parser.error("Task, policy, and checkpoint routing are fixed by the manifest.")
    rows = list(tasks.values()) if args.task == "all" else [tasks[args.task]]
    for row in rows:
        checkpoint = args.checkpoint_root.resolve() / Path(row["local_checkpoint_dir"]).name
        if args.action == "list":
            print(f'{row["task"]}: {row["policy"]} | {row["hf_url"]}')
            continue
        if args.action == "download":
            command = ["hf", "download", row["hf_repo_id"], "--repo-type", "model",
                       "--revision", row["hf_revision"], "--local-dir", str(checkpoint)]
            if not args.dry_run and shutil.which("hf") is None:
                parser.error("Install huggingface_hub[cli] in your policy environment first.")
        else:
            wrapper = ROOT / "policy" / row["policy"] / "eval.sh"
            if not wrapper.is_file():
                parser.error(f"Official evaluation wrapper is missing: {wrapper}")
            command = ["bash", str(wrapper), row["task"], args.setting,
                       str(checkpoint), args.gpu, "--pytorch_device", "cuda",
                       "--headless", "True", "--max_episodes", str(args.episodes)]
            if row["policy"] == "dp":
                command += ["--dp_num_inference_steps", "10"]
            if row["policy"] == "smolvla":
                command += ["--smolvla_rescale_gripper", "auto"]
            command += extra
            if not args.dry_run and not (checkpoint / "config.json").is_file():
                parser.error(f"Checkpoint missing: run download {row['task']} first.")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
