"""One in-memory replay ring per source, served over a private Unix socket."""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import random
import resource
import socket
import socketserver
import struct
from threading import Lock


CAPACITY = 500_000
SOURCES = ("rgb-count", "rgb-hybrid", "edges-count", "edges-hybrid")
MAX_PACKET = 32 * 2**20


def encode_bytes(value):
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    raise TypeError(type(value))


def decode_bytes(value):
    return (
        base64.b64decode(value["bytes"], validate=True)
        if set(value) == {"bytes"}
        else value
    )


def send(stream, value):
    blob = json.dumps(
        value, default=encode_bytes, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(blob) > MAX_PACKET:
        raise ValueError("Replay message too large")
    stream.sendall(struct.pack("!I", len(blob)) + blob)


def receive(reader):
    header = reader.read(4)
    if len(header) != 4:
        raise EOFError("Replay socket was closed")
    (size,) = struct.unpack("!I", header)
    if not 0 < size <= MAX_PACKET:
        raise ValueError("Invalid replay message length")
    blob = reader.read(size)
    if len(blob) != size:
        raise EOFError("Incomplete replay message")
    return json.loads(blob, object_hook=decode_bytes)


class ReplayClient:
    def __init__(self, path, source):
        self.source = source
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(45)
        self.socket.connect(str(path))
        self.reader = self.socket.makefile("rb")

    def request(self, command, **payload):
        send(self.socket, dict(command=command, source=self.source, **payload))
        reply = receive(self.reader)
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply["value"]

    def close(self):
        self.reader.close()
        self.socket.close()


def vector(value, size):
    if len(value) != size or not all(
        type(x) in (int, float) and math.isfinite(x) for x in value
    ):
        raise ValueError("Invalid replay vector")
    return tuple(value)


def observation(value):
    images, numeric, base = value
    if len(images) != 2 or not all(
        isinstance(x, bytes)
        and 4 <= len(x) <= 64 * 1024
        and x[:2] == b"\xff\xd8"
        and x[-2:] == b"\xff\xd9"
        for x in images
    ):
        raise ValueError("Expected two JPEG replay frames")
    return tuple(images), vector(numeric, 18), vector(base, 6)


class SourceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rows = []
        self.offset = 0
        self.sequence = 0
        self.last = None

    def close_interrupted_episode(self):
        if self.last is not None:
            index = (
                (self.offset - 1) % len(self.rows)
                if len(self.rows) == self.capacity
                else len(self.rows) - 1
            )
            self.rows[index] = (*self.rows[index][:-1], True)
            self.last = None
        return self.sequence

    def append(
        self,
        sequence,
        previous,
        following,
        action,
        rewards,
        discount,
        terminated,
        truncated,
    ):
        if self.sequence and sequence == self.sequence - 1:
            return (
                self.sequence
            )  # A reconnect can retry the last acknowledged insertion.
        if sequence != self.sequence:
            raise ValueError("Out-of-order replay insertion")
        if any(type(flag) is not bool for flag in (terminated, truncated)) or (
            terminated and truncated
        ):
            raise ValueError("Invalid termination flags")
        if (
            type(discount) not in (int, float)
            or not 0 <= discount <= 1
            or (discount == 0) != terminated
        ):
            raise ValueError("Invalid bootstrap discount")
        if self.last is None and previous is None:
            raise ValueError("Episode start requires a first observation")
        if self.last is not None and previous is not None:
            raise ValueError("Previous episode is still open")
        previous = observation(previous) if previous is not None else self.last
        following = observation(following)
        row = (
            sequence,
            previous,
            following,
            vector(action, 6),
            vector(rewards, 2),
            discount,
            terminated,
            truncated,
        )
        if len(self.rows) == self.capacity:
            self.rows[self.offset] = row
            self.offset = (self.offset + 1) % self.capacity
        else:
            self.rows.append(row)
        self.last = None if terminated or truncated else following
        self.sequence += 1
        return self.sequence


class ReplayPool:
    def __init__(self, capacity=CAPACITY):
        if capacity < len(SOURCES) or capacity % len(SOURCES):
            raise ValueError("Capacity must divide evenly across sources")
        self.capacity = capacity
        self.sources = {
            name: SourceReplay(capacity // len(SOURCES)) for name in SOURCES
        }
        self.lock = Lock()

    def dispatch(self, request):
        name = request.pop("source")
        command = request.pop("command")
        current = self.sources[name]
        with self.lock:
            if command == "append":
                return current.append(**request)
            if command == "stats":
                return dict(
                    capacity=self.capacity,
                    counts={key: len(part.rows) for key, part in self.sources.items()},
                    next_sequence=current.sequence,
                    episode_open=current.last is not None,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                )
            if command == "close_interrupted_episode":
                return current.close_interrupted_episode()
            if command != "sample":
                raise ValueError("Unknown replay operation")
            return self.sample(name, **request)

    def sample(self, source, batch, seed):
        if batch not in (128, 256):
            raise ValueError("Unsupported replay batch")
        rng = random.Random(seed)
        others = [name for name in SOURCES if name != source]
        rng.shuffle(others)
        half = batch // 2
        base, remainder = divmod(half, len(others))
        quotas = {
            source: half,
            **{name: base + (i < remainder) for i, name in enumerate(others)},
        }
        if any(len(self.sources[name].rows) < count for name, count in quotas.items()):
            raise ValueError("Replay sources have not reached the sampling minimum")
        rows = [
            (name, row)
            for name, count in quotas.items()
            for row in rng.sample(self.sources[name].rows, count)
        ]
        rng.shuffle(rows)
        return rows


def serve(path):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                try:
                    value = self.server.pool.dispatch(receive(self.rfile))
                    send(self.request, {"value": value})
                except (EOFError, OSError):
                    return
                except (KeyError, TypeError, ValueError) as error:
                    send(self.request, {"error": str(error)})

    os.umask(0o077)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socketserver.ThreadingUnixStreamServer(str(path), Handler) as server:
        server.daemon_threads = True
        server.pool = ReplayPool()
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("socket", type=Path)
    args = parser.parse_args()
    serve(args.socket)
