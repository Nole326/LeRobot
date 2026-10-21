"""Re-exec a starter CLI inside the course LeIsaac container."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from starter.config import PROJECT_ROOT

LEISAAC_IMAGE = os.environ.get("LEISAAC_IMAGE", "isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src")
# Set only by the docker handoff below, so a relaunch cannot start another container.
REEXEC_ENV = "STARTER_CONTAINER_REEXEC"
CACHE_SUBDIRS = ("kit", "ov", "pip", "glcache", "computecache", "logs", "data", "documents")


def in_course_container() -> bool:
    """True when this process is inside the course image layout."""
    return Path("/isaac-sim/python.sh").is_file() and Path("/leisaac/source/leisaac").is_dir()


def isaac_packages_present() -> bool:
    """True when this interpreter can resolve Isaac Lab and LeIsaac."""
    import importlib.util

    try:
        return importlib.util.find_spec("isaaclab") is not None and importlib.util.find_spec("leisaac") is not None
    except (ImportError, ValueError):
        return False


def isaac_runtime_ready() -> bool:
    return in_course_container() or isaac_packages_present()


def _cache_root() -> Path:
    configured = os.environ.get("ISAAC_CACHE")
    if not configured:
        raise RuntimeError(
            "Automatic Docker startup requires ISAAC_CACHE. Set it to the cache "
            "directory prepared in the README, or run in an installed Isaac environment."
        )
    root = Path(configured).expanduser().resolve()
    missing = [name for name in CACHE_SUBDIRS if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(
            f"ISAAC_CACHE={root} is missing directories: {', '.join(missing)}. "
            "Create the eight subdirectories shown in the README before starting Docker."
        )
    return root


def ensure_in_leisaac_container() -> None:
    """Stay in this process when LiftCube can run here; otherwise re-exec in the course image.

    A local install is detected by the presence of ``isaaclab`` and ``leisaac`` on
    ``sys.path``. That path does not claim to be the course container.
    Library calls and tests pass an explicit argv into ``main`` and do not get here.
    A real ``python -m starter...`` invocation does.
    """
    if isaac_runtime_ready():
        return
    if os.environ.get(REEXEC_ENV) == "1":
        raise RuntimeError(
            "LiftCube runtime is unavailable after the Docker handoff: "
            "expected the course image layout or importable isaaclab and leisaac."
        )
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    module = getattr(spec, "name", None)
    if not module or not module.startswith("starter."):
        raise RuntimeError("Container launch only supports `python -m starter...` entry points.")
    starter = PROJECT_ROOT.resolve()
    repository = starter.parent
    cache = _cache_root()
    inner = " ".join(
        [
            f"export {REEXEC_ENV}=1",
            "PYTHONPATH=" + shlex.quote(f"{starter}/src:/leisaac/source/leisaac") + ";",
            "cd",
            shlex.quote(str(starter)) + ";",
            "/isaac-sim/python.sh",
            "-m",
            shlex.quote(module),
            *(shlex.quote(arg) for arg in sys.argv[1:]),
        ]
    )
    name = f"starter-liftcube-{os.getpid()}"
    docker_args = [
        "run",
        "--rm",
        "--name",
        name,
        "--runtime=nvidia",
        "--gpus",
        "device=0",
        "--device",
        "/dev/nvidiactl",
        "--device",
        "/dev/nvidia0",
        "--device",
        "/dev/nvidia-modeset",
        "--device",
        "/dev/nvidia-uvm",
        "--device",
        "/dev/nvidia-uvm-tools",
        "--device",
        "/dev/nvidia-caps/nvidia-cap1",
        "--device",
        "/dev/nvidia-caps/nvidia-cap2",
        "--device",
        "/dev/dri/card1",
        "--device",
        "/dev/dri/renderD128",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=0",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "-e",
        f"{REEXEC_ENV}=1",
        "-v",
        f"{cache}/kit:/isaac-sim/kit/cache:rw",
        "-v",
        f"{cache}/ov:/root/.cache/ov:rw",
        "-v",
        f"{cache}/pip:/root/.cache/pip:rw",
        "-v",
        f"{cache}/glcache:/root/.cache/nvidia/GLCache:rw",
        "-v",
        f"{cache}/computecache:/root/.nv/ComputeCache:rw",
        "-v",
        f"{cache}/logs:/root/.nvidia-omniverse/logs:rw",
        "-v",
        f"{cache}/data:/root/.local/share/ov/data:rw",
        "-v",
        f"{cache}/documents:/root/Documents:rw",
        "-v",
        f"{repository}:{repository}",
        "-w",
        str(starter),
        "--entrypoint",
        "bash",
        LEISAAC_IMAGE,
        "-lc",
        inner,
    ]
    if _docker_socket_open():
        os.execvp("docker", ["docker", *docker_args])
    os.execvp("sg", ["sg", "docker", "-c", " ".join(shlex.quote(part) for part in ["docker", *docker_args])])


def _docker_socket_open() -> bool:
    try:
        result = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return False
    return result.returncode == 0
