"""Inspect an experiment plan without importing a simulator or learning library.

Exit 0: specified configuration (NOT training acceptance); 2: invalid; 3: incomplete.
No environment creation, downloads, training, or filesystem writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

FIELDS = {
    "policy": {"name", "image_keys", "proprio_names", "observation_schema_id",
               "task_condition", "chunk_size", "execute_steps"},
    "control": {"action_schema_id", "representation", "frame", "components", "units",
                "lower", "upper", "physics_dt_s", "physics_steps_per_action", "policy_hz"},
    "data": {"action_schema_id", "observation_schema_id", "split_unit", "statistics_split"},
    "training": {"phase", "seed", "budget", "budget_unit"},
    "evaluation": {"protocol_id", "max_decisions", "seed_manifest_sha256"},
}
JOINTS = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect(plan):
    """Check declared metadata only; schemas do not establish sensor provenance."""
    errors, pending = [], []
    if not isinstance(plan, dict):
        return {"errors": ["Plan must be a mapping"], "pending": []}
    if set(plan) != {"schema_version", *FIELDS}:
        errors.append("Missing/unknown top-level fields")
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        errors.append("schema_version must be integer 1")
    for section, fields in FIELDS.items():
        if not isinstance(plan.get(section), dict) or set(plan[section]) != fields:
            errors.append(f"{section}: missing/unknown fields")
    if errors:
        return {"errors": errors, "pending": pending}

    p, c, d, t, e = (plan[k] for k in FIELDS)

    def text(value, key, choices=None, required=True):
        if not isinstance(value, str):
            errors.append(f"{key}: expected text")
        elif not value.strip():
            if required:
                pending.append(key)
        elif choices is not None and value not in choices:
            errors.append(f"{key}: unsupported value {value!r}")

    def count(value, key, allow_zero=False):
        if type(value) is not int or value < 0:
            errors.append(f"{key}: expected non-negative integer")
        elif value == 0 and not allow_zero:
            pending.append(key)

    def names(value, key):
        valid = isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
        if not valid or len(value) != len(set(value)):
            errors.append(f"{key}: expected unique non-empty strings")
            return False
        if not value:
            pending.append(key)
        return True

    text(p["name"], "policy.name", {"act", "smolvla"})
    if names(p["image_keys"], "policy.image_keys"):
        if any(not re.fullmatch(r"observation\.images\.[a-zA-Z0-9_]+", x) for x in p["image_keys"]):
            errors.append("policy.image_keys: expected explicit image feature keys")
    if names(p["proprio_names"], "policy.proprio_names"):
        if any(x not in JOINTS for x in p["proprio_names"]):
            errors.append("policy.proprio_names: unknown field; extend the reviewed proprioception whitelist first")
    for key in ("observation_schema_id", "task_condition"):
        text(p[key], f"policy.{key}")
    for key in ("chunk_size", "execute_steps"):
        count(p[key], f"policy.{key}")
    if all(type(p[k]) is int for k in ("chunk_size", "execute_steps")):
        if p["execute_steps"] > p["chunk_size"]:
            errors.append("policy.execute_steps exceeds chunk_size")

    text(c["action_schema_id"], "control.action_schema_id")
    text(c["representation"], "control.representation", {"absolute_joint", "delta_ee"})
    text(c["frame"], "control.frame", {"joint", "table", "robot_base"})
    components_valid = names(c["components"], "control.components")
    for key in ("units", "lower", "upper"):
        if not isinstance(c[key], list):
            errors.append(f"control.{key}: expected list")
    arrays = components_valid and all(isinstance(c[k], list) for k in ("units", "lower", "upper"))
    if arrays:
        if len({len(c[k]) for k in ("components", "units", "lower", "upper")}) != 1:
            errors.append("control: component/unit/limit lengths differ")
        for unit in c["units"]:
            if not isinstance(unit, str) or unit not in {"rad", "m", "normalized"}:
                errors.append("control.units: use explicit rad/m/normalized units")
        for low, high in zip(c["lower"], c["upper"]):
            if not number(low) or not number(high) or low >= high:
                errors.append("control: bounds must be finite with lower < upper")
        if c["representation"] == "absolute_joint":
            if c["frame"] != "joint" or c["components"] != JOINTS or c["units"] != ["rad"] * 6:
                errors.append("absolute_joint expects ordered SO-101 joint targets in radians, frame=joint")
        if c["representation"] == "delta_ee" and c["frame"] == "joint":
            errors.append("delta_ee requires a Cartesian frame, not joint")
        if c["representation"] == "delta_ee":
            permitted = {"dx": "m", "dy": "m", "dz": "m", "droll": "rad",
                         "dpitch": "rad", "dyaw": "rad", "gripper": "normalized"}
            if any(name not in permitted or unit != permitted[name]
                   for name, unit in zip(c["components"], c["units"])):
                errors.append("delta_ee: invalid component/unit pair")
    for key in ("physics_dt_s", "policy_hz"):
        if not number(c[key]) or c[key] < 0:
            errors.append(f"control.{key}: expected finite non-negative number")
        elif c[key] == 0:
            pending.append(f"control.{key}")
    count(c["physics_steps_per_action"], "control.physics_steps_per_action")
    timing = [c[k] for k in ("physics_dt_s", "physics_steps_per_action", "policy_hz")]
    if all(number(x) and x > 0 for x in timing) and not math.isclose(math.prod(timing), 1.0, rel_tol=1e-6):
        errors.append("control: policy_hz * physics_dt_s * physics_steps_per_action must equal 1")

    for key in ("action_schema_id", "observation_schema_id"):
        text(d[key], f"data.{key}")
    for key, choices in (("split_unit", {"episode"}), ("statistics_split", {"train"})):
        text(d[key], f"data.{key}", choices)
    if d["action_schema_id"] and c["action_schema_id"] and d["action_schema_id"] != c["action_schema_id"]:
        errors.append("data/control action schemas differ: conversion must be explicit and validated")
    if d["observation_schema_id"] and p["observation_schema_id"] and d["observation_schema_id"] != p["observation_schema_id"]:
        errors.append("data/policy observation schemas differ")
    text(t["phase"], "training.phase", {"offline_bc", "online_rl"})
    text(t["budget_unit"], "training.budget_unit", {"updates", "environment_steps"})
    count(t["seed"], "training.seed", allow_zero=True)
    count(t["budget"], "training.budget")
    expected_unit = {"offline_bc": "updates", "online_rl": "environment_steps"}.get(
        t["phase"] if isinstance(t["phase"], str) else "")
    if expected_unit and t["budget_unit"] != expected_unit:
        errors.append("training: phase/budget unit mismatch")
    text(e["protocol_id"], "evaluation.protocol_id")
    text(e["seed_manifest_sha256"], "evaluation.seed_manifest_sha256")
    count(e["max_decisions"], "evaluation.max_decisions")
    if isinstance(e["seed_manifest_sha256"], str) and e["seed_manifest_sha256"]:
        if not re.fullmatch(r"[0-9a-f]{64}", e["seed_manifest_sha256"]):
            errors.append("evaluation.seed_manifest_sha256: expected lowercase SHA256")
    return {"errors": errors, "pending": pending}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.config.read_bytes()
        plan = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"Cannot read plan: {error}\n")
    report = inspect(plan)
    code = 2 if report["errors"] else 3 if report["pending"] else 0
    report.update(configuration_status={0: "specified", 2: "invalid", 3: "incomplete"}[code],
                  training_readiness="NOT ASSESSED: no runtime/evidence validation or trainer launch",
                  config_sha256=hashlib.sha256(raw).hexdigest())
    if args.print_config:
        report["config"] = plan
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
