"""Training curves; independent of inference exports and resume checkpoints."""
from pathlib import Path


def open_writer(config):
    if not config.get("logging", {}).get("tensorboard", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required for training curves; install the logging extra in the approved environment") from exc
    return SummaryWriter(log_dir=str(Path(config.get("logging", {}).get("dir", "outputs")) / "tensorboard"))


def log_row(writer, stage, row, step):
    if writer is not None:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{stage}/{key}", value, step)
        writer.flush()
