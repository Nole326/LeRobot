"""Local adaptation: refresh RGB after reset without advancing physics.

Uses the render-only procedure already checked in the course pi05 diagnostics.
"""
import time
import numpy as np


def prepare_front_rgb(raw, observation, camera):
    from isaacsim.core.utils.stage import is_stage_loading

    sensor = raw.scene.sensors[camera]
    previous = None
    stable = 0
    started = time.monotonic()
    for frame in range(300):
        if time.monotonic() - started > 180:
            raise RuntimeError("RGB preparation timed out before policy execution")
        raw.sim.render()
        sensor.reset()  # Invalidate only the sensor's bookkeeping, not the scene.
        sensor.update(0.0, force_recompute=True)
        rgb = sensor.data.output["rgb"][..., :3].clone()
        pixels = rgb[0].cpu().numpy()
        change = None if previous is None else float(np.abs(pixels.astype(float) - previous).mean())
        stable = stable + 1 if frame >= 30 and not is_stage_loading() and change is not None and change < 1.5 else 0
        previous = pixels.astype(float)
        if stable >= 10:
            if not np.isfinite(pixels).all() or float(pixels.std()) <= 10:
                raise RuntimeError("Invalid RGB after render preparation")
            observation["policy"][camera] = rgb
            return observation, {"render_frames": frame + 1, "last_mean_change": change,
                                 "elapsed_s": time.monotonic() - started, "physics_steps": 0}
    raise RuntimeError("RGB did not stabilize before policy execution")
