"""Camera sampling, reproducible checkpoints, and complete run metadata."""

import json
import os
import random
import subprocess

import numpy as np
import torch


class CameraSampler:
    def __init__(self, cameras):
        self.pool = list(cameras)
        self.stack = self.pool.copy()

    def sample(self):
        if not self.stack:
            self.stack = self.pool.copy()
        if not self.stack:
            return None
        return self.stack.pop(random.randint(0, len(self.stack) - 1))

    def state_dict(self):
        return {name: [cam.image_name for cam in getattr(self, name)] for name in ("pool", "stack")}

    def load_state_dict(self, state):
        cameras = {cam.image_name: cam for cam in self.pool}
        if set(state["pool"]) != set(cameras) or len(state["pool"]) != len(self.pool):
            raise ValueError("Checkpoint camera pool differs from the current dataset")
        self.pool = [cameras[name] for name in state["pool"]]
        self.stack = [cameras[name] for name in state["stack"]]


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def run_config(dataset, opt, pipe, run_args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"schema_version": 1, "model": vars(dataset).copy(),
            "optimization": vars(opt).copy(), "pipeline": vars(pipe).copy(),
            "run": dict(run_args), "git_commit": revision, "git_dirty": dirty,
            "torch_version": str(torch.__version__)}


def save_run_config(path, config):
    with open(path, "w") as file:
        json.dump(config, file, indent=2)


def save_checkpoint(path, state):
    # Avoid replacing a usable checkpoint with a partially written file.
    temporary = path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, map_location=None):
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("This checkpoint lacks complete joint training state. Legacy background-only checkpoints cannot resume joint training.")
    return state


def parse_training_args(parser, argv):
    """Restore saved defaults; explicit command-line arguments still win."""
    args = parser.parse_args(argv)
    if args.start_checkpoint:
        config = load_checkpoint(args.start_checkpoint, map_location="cpu")["config"]
        defaults = {}
        for section in ("model", "optimization", "pipeline", "run"):
            defaults.update(config[section])
        known = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})
        args = parser.parse_args(argv)
    return args
