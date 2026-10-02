"""CPU-only inventory; do not import Isaac/initialize CUDA here."""
import importlib.metadata as md
import json
import platform
import sys

packages = ["torch", "torchvision", "numpy", "gymnasium", "isaaclab", "isaacsim", "lerobot", "leisaac", "av", "torchcodec", "protobuf", "wandb", "opencv-python", "opencv-python-headless"]
versions = {}
for name in packages:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "versions": versions}, indent=2))
