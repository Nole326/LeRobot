"""Bounded Linux-local RGB transport. Pixels never enter a disk file."""

import hashlib
import os
from pathlib import Path
import socket
import struct
import time

import numpy as np

IMAGE_BYTES = 480 * 640 * 3
PAYLOAD_BYTES = IMAGE_BYTES * 2 + 6 * 8
HEADER = struct.Struct("!QQ")


def address(out):
    # Abstract Unix sockets use kernel memory, not filesystem image files.
    return (
        "\0lerobot-rgb-"
        + hashlib.sha256(os.fsencode(Path(out).resolve())).hexdigest()[:40]
    )


def validate(data):
    if set(data) != {"front", "wrist", "joint_motor"}:
        raise ValueError("RGB/proprioception whitelist mismatch")
    for name in ("front", "wrist"):
        if data[name].shape != (1, 480, 640, 3) or data[name].dtype != np.uint8:
            raise ValueError("Expected real uint8 dual RGB")
    if (
        data["joint_motor"].shape != (1, 6)
        or not np.isfinite(data["joint_motor"]).all()
    ):
        raise ValueError("Invalid joints")


def read_exact(peer, size):
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        count = peer.recv_into(view[offset:])
        if not count:
            raise EOFError("Peer closed an incomplete observation")
        offset += count
    return buffer


def send_observation(out, query, data, timeout=180):
    validate(data)
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(timeout)
        while True:
            try:
                peer.connect(address(out))
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if (Path(out) / "model-error.json").exists():
                    raise RuntimeError("Model failed before receiving image")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model observation receiver unavailable")
                time.sleep(0.05)
        peer.sendall(HEADER.pack(query, PAYLOAD_BYTES))
        for name in ("front", "wrist"):
            peer.sendall(np.ascontiguousarray(data[name]).tobytes())
        peer.sendall(np.asarray(data["joint_motor"], dtype="<f8").tobytes())
        if read_exact(peer, 1) != b"\x01":
            raise ValueError("Observation not acknowledged")


def receive_observation(out, query, timeout=180):
    deadline = time.monotonic() + timeout
    error = Path(out) / "sim-error.json"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address(out))
        listener.listen(1)
        while True:
            if error.exists():
                raise RuntimeError("Simulator failed; see sim-error.json")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Simulator observation unavailable")
            listener.settimeout(min(1.0, remaining))
            try:
                peer, _ = listener.accept()
                break
            except socket.timeout:
                continue
        with peer:
            peer.settimeout(timeout)
            actual_query, size = HEADER.unpack(read_exact(peer, HEADER.size))
            if actual_query != query or size != PAYLOAD_BYTES:
                raise ValueError("Observation sequence/size mismatch")
            buffer = read_exact(peer, size)
            data = {
                name: np.frombuffer(
                    buffer, np.uint8, IMAGE_BYTES, i * IMAGE_BYTES
                ).reshape(1, 480, 640, 3)
                for i, name in enumerate(("front", "wrist"))
            }
            data["joint_motor"] = np.frombuffer(
                buffer, "<f8", 6, 2 * IMAGE_BYTES
            ).reshape(1, 6)
            validate(data)
            peer.sendall(b"\x01")
            return data
