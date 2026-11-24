"""Train PickOrange ACT on the original dual-camera qpos or segmented EE archives."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np

from .data import Episodes, decode


def train(args, data, log_update):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cams = ("observation.images.front", "observation.images.wrist")
    inputs = {cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for cam in cams}
    inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(len(data.amu),))}
    config = ACTConfig(
        input_features=inputs,
        output_features=outputs,
        chunk_size=15,
        n_action_steps=1,
        temporal_ensemble_coeff=0.001,
        optimizer_lr=1e-4,
        optimizer_lr_backbone=1e-5,
        optimizer_weight_decay=1e-4,
        kl_weight=10.0,
        device=args.device,
    )
    policy = ACTPolicy(config).to(args.device).train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    mean = torch.tensor(rgb_mean, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(rgb_std, device=args.device).view(1, 3, 1, 1)
    normalization = {k: getattr(data, k).tolist() for k in ("qmu", "qstd", "amu", "astd")}
    execution = {
        "mode": "absolute_ee_robot_base" if data.ee else "stride2",
        "physics_hz": 60,
        "action_hz": 30,
        "replan_physics_ticks": 2,
        "chunk": 15,
        "ensemble": 0.001,
        "image_order": ["front", "wrist"],
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "camera_shape": [480, 640, 3],
    }
    if data.ee:
        execution["skill_end"] = data.single
    if data.single:
        from .ee_loss import segmented_loss

        execution.update(skill_end=True, skill_end_encoding="logit")
    data.cache()
    milestones = {10_000, 20_000, 30_000, 40_000, 50_000, 65_000, 80_000, 100_000, 120_000, 160_000, args.updates}
    with ThreadPoolExecutor(max_workers=args.microbatch) as pool:
        for step in range(1, args.updates + 1):
            samples = data.choose(rng, args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for offset in range(0, args.batch_size, args.microbatch):
                items = pool.map(data.sample, samples[offset : offset + args.microbatch])
                q, a, pad, front, wrist = (
                    torch.from_numpy(np.stack(x)).to(args.device) for x in zip(*items)
                )
                batch = {"observation.state": q, "action": a, "action_is_pad": pad}
                for cam, rgb in zip(cams, (front, wrist)):
                    batch[cam] = (rgb.permute(0, 3, 1, 2).float().div_(255) - mean) / std
                loss, _ = (
                    segmented_loss(policy, batch, data.stop_weight)
                    if data.single
                    else policy(batch)
                )
                weighted = loss * (args.microbatch / args.batch_size)
                weighted.backward()
                total_loss += weighted.detach().item()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            if step <= 3 or step % 25 == 0 or step == args.updates:
                log_update(step, total_loss)
            if step in milestones:
                path = args.output / f"checkpoint-{step}"
                tmp = path.with_suffix(".partial")
                policy.save_pretrained(tmp)
                for name, values in (
                    ("normalization", normalization),
                    ("execution", dict(execution, optimizer_updates=step)),
                ):
                    (tmp / (name + ".json")).write_text(
                        json.dumps(values, indent=2), encoding="utf-8"
                    )
                tmp.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qpos", "ee", "ee-full"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1704)
    parser.add_argument(
        "--check-data", action="store_true", help="Validate numeric labels and both video streams"
    )
    args = parser.parse_args()
    if args.updates < 1 or (not args.check_data and args.output is None):
        parser.error("Training requires --output and a positive update count")
    if args.batch_size < 1 or args.batch_size % args.microbatch:
        parser.error("--batch-size must be a positive multiple of --microbatch")
    data = Episodes(args.manifest, args.mode)
    if args.check_data:
        for q, _, paths in data.rows:
            for path in paths:
                decode(path, len(q), cache=False)
        print(f"{len(data.rows)} episodes, {len(data.intervals)} sampling intervals")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "liftcube/src"))
    from starter.logging.swanlab import ScalarWriter

    args.output.mkdir(parents=True, exist_ok=False)
    tracking = {
        "seed": args.seed,
        "data": {"batch_size": args.batch_size},
        "logging": {"dir": str(args.output), "swanlab": {"name": f"pickorange-act-{args.mode}"}},
    }
    with (
        closing(ScalarWriter(tracking, "pickorange-act")) as writer,
        (args.output / "loss.jsonl").open("x", encoding="utf-8") as metrics,
    ):
        def log_update(step, loss):
            metrics.write(json.dumps({"step": step, "loss": loss}) + "\n")
            metrics.flush()
            writer.log({"loss": loss}, step, prefix="train")

        train(args, data, log_update)


if __name__ == "__main__":
    main()
