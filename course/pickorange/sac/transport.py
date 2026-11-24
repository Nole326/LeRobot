"""Bounded ACT inference messages over one local stream."""

import json
import socket
import struct
from pathlib import Path

import numpy as np

IMAGE_BYTES = 480 * 640 * 3
PAYLOAD_BYTES = IMAGE_BYTES * 2 + 6 * 8
HEADER = struct.Struct("!QQ")


def address(out):
    return str(Path(out).resolve() / "rgb.sock")


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


LENGTH = struct.Struct("!I")
MAX_FEEDBACK = 256 * 1024


def listener(out):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(900)
    server.bind(address(out))
    server.listen(1)
    return server


def connect(out):
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    peer.settimeout(900)
    peer.connect(address(out))
    return peer


def send_observation(peer, query, obs):
    validate(obs)
    peer.sendall(HEADER.pack(query, PAYLOAD_BYTES))
    for name in ("front", "wrist"):
        peer.sendall(np.ascontiguousarray(obs[name]).tobytes())
    peer.sendall(np.asarray(obs["joint_motor"], dtype="<f8").tobytes())


def receive_observation(peer, query):
    if HEADER.unpack(read_exact(peer, HEADER.size)) != (query, PAYLOAD_BYTES):
        raise ValueError("Unexpected ACT observation header")
    data = read_exact(peer, PAYLOAD_BYTES)
    return dict(
        front=np.frombuffer(data, np.uint8, IMAGE_BYTES).reshape(1, 480, 640, 3),
        wrist=np.frombuffer(data, np.uint8, IMAGE_BYTES, IMAGE_BYTES).reshape(
            1, 480, 640, 3
        ),
        joint_motor=np.frombuffer(data, "<f8", 6, 2 * IMAGE_BYTES).reshape(1, 6),
    )


def send_action(peer, actions):
    peer.sendall(np.asarray(actions, dtype="<f4").tobytes())


def receive_action(peer, stride):
    return (
        np.frombuffer(read_exact(peer, stride * 6 * 4), "<f4").reshape(stride, 6).copy()
    )


def send_feedback(peer, row):
    data = json.dumps(row, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_FEEDBACK:
        raise ValueError("Oversize ACT feedback")
    peer.sendall(LENGTH.pack(len(data)) + data)


def receive_feedback(peer):
    (size,) = LENGTH.unpack(read_exact(peer, LENGTH.size))
    if not 0 < size <= MAX_FEEDBACK:
        raise ValueError("Unexpected ACT feedback length")
    return json.loads(read_exact(peer, size))
