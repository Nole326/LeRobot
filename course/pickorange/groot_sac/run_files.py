"""Ownership and failure reporting for paired GR00T processes."""

from contextlib import contextmanager
import traceback

from ..common import publish


@contextmanager
def run_directory(out, *, simulator=False):
    if simulator:
        (out / "simulation.claim").open("x").close()
        error = out / "sim-error.json"
    else:
        out.mkdir(parents=True, exist_ok=False)
        error = out / "model-error.json"
    try:
        yield
    except BaseException:
        publish(error, {"traceback": traceback.format_exc()})
        raise
