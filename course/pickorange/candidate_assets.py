"""Fixed public inference exports; never fetch or execute model-repository code."""
import hashlib

from common import sha256

SMOL = {
    'config.json': [2275, '4d814e8818739d0ecdb6ec7c98aa4ac8f8add85d'],
    'model.safetensors': [1197789224, '8ec908fd230780b76cc983425744224699f38a1d1fd4541d4b6afd4b4439dc9b'],
    'policy_preprocessor.json': [1717, '4175f3d691a20f30894c2d0426583c1a6f45e5e9'],
    'policy_postprocessor.json': [660, '6f8997e5be18c67bad9377dd2cd9622ba38b5ae3'],
    'policy_preprocessor_step_5_normalizer_processor.safetensors':
        [3756, '0b9214914398156f17ddf08a5082c20b24efb70a7ba7677e8bc3d3ff68a722db'],
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors':
        [3756, '0b9214914398156f17ddf08a5082c20b24efb70a7ba7677e8bc3d3ff68a722db'],
}

N16 = {
    "config.json": [
        1840,
        "7c9f5d8b71129376f5d330ab470af1c1f011e2b7"
    ],
    "embodiment_id.json": [
        191,
        "9db6cf1b1236298d5de030192d4d9c3f1d143e4c"
    ],
    "model-00001-of-00002.safetensors": [
        4996585112,
        "d91f41ca2d048e50a0eb9bbd7b6bdc78889d26dbd3311aea8593b4102f6a47dc"
    ],
    "model-00002-of-00002.safetensors": [
        4414037400,
        "a794be81fdd8a0403711d98749cd975c8a3634b4f7fbbe350755487b7af6f430"
    ],
    "model.safetensors.index.json": [
        121820,
        "95a5b50e82c256f8720939536ec172b289e4b483"
    ],
    "processor_config.json": [
        11088,
        "1442fd100a6ed5c1d60da16d4007570b746fb613"
    ],
    "statistics.json": [
        283623,
        "f42d3d72473d0fcd35557642adef58796d9ffc37"
    ]
}
PI05 = {
    "config.json": [
        2444,
        "2294634d3e8675a16ffb84210157c5d60e6c825a"
    ],
    "model.safetensors": [
        9354050752,
        "d904bf245b78532f846de406da36be3493cb230fac9f7102813fa0b9e0a06d15"
    ],
    "policy_postprocessor.json": [
        779,
        "15fc6d1e912a1bf28d26b7708036f04b7858be92"
    ],
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": [
        7552,
        "79ccc30dfce69479e2e2a40eea5dbcab4809fd69d2f009da577d4a5462968fbb"
    ],
    "policy_preprocessor.json": [
        1927,
        "d659f45a1dc6e315b6f48ac0c8f73d9ff3d3b3ff"
    ],
    "policy_preprocessor_step_3_normalizer_processor.safetensors": [
        7560,
        "73bf6959102b15634e7b00cc0a831ac0a93b85fb2fa491cc76bef16ec50b40f9"
    ]
}


def verify_files(folder, expected):
    result = {}
    for name, (size, digest) in expected.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'Missing or wrong-sized checkpoint file: {name}')
        if len(digest) == 40:
            actual = hashlib.sha1(f'blob {size}\0'.encode() + path.read_bytes()).hexdigest()
        else:
            actual = sha256(path)
        if actual != digest:
            raise ValueError(f'Checkpoint digest mismatch: {name}')
        result[name] = digest
    return result
