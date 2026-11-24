"""Export the actor and encoder from a trusted historical training checkpoint."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ...common import publish


def export(path, output):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = state["config"]
    if config.get("reward") == "retained_orange_tiers":
        release = dict(
            model="Residual-4",
            encoder="dual_rgb_resblocks",
            base_subfolder="ACT/ACT-1/100k",
            evaluation=dict(
                synchronized_reset=True, fixed_gripper_effort_nm=10, num_envs=1
            ),
        )
    elif config.get("encoder") in ("edges", "rgb") and config["reward"] in (
        "hybrid",
        "count",
    ):
        release = dict(
            encoder=config["encoder"],
            reward=config["reward"],
            base_subfolder="ACT/ACT-1/80k",
            evaluation=(
                {"protocol": "independent_single_scene", "num_envs": 1}
                if config["encoder"] == "edges"
                and config["reward"] == "hybrid"
                and state["actor_updates"] == 26000
                else {"synchronized_reset": True}
            ),
        )
    else:
        raise ValueError("Unrecognized training checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    save_file(
        {
            f"{module}.{name}": value.contiguous()
            for module in ("encoder", "actor")
            for name, value in state[module].items()
        },
        str(output / "model.safetensors"),
    )
    publish(
        output / "config.json", dict(**release, actor_updates=state["actor_updates"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
