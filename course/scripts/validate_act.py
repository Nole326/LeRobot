"""Synthetic CUDA/ACT/video smoke. No robot, dataset training, or W&B run."""
import json
from pathlib import Path
import subprocess
import time
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

torch.set_num_threads(4)
torch.manual_seed(1)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
cfg = ACTConfig(
    input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        'observation.images.front': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        'observation.images.wrist': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    },
    output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    pretrained_backbone_weights=None, chunk_size=4, n_action_steps=1,
    dim_model=128, n_heads=4, dim_feedforward=256, n_encoder_layers=1,
    n_decoder_layers=1, n_vae_encoder_layers=1, device='cuda',
)
policy = ACTPolicy(cfg).cuda()
batch = {
    'observation.state': torch.zeros(2, 6, device='cuda'),
    'observation.images.front': torch.rand(2, 3, 64, 64, device='cuda'),
    'observation.images.wrist': torch.rand(2, 3, 64, 64, device='cuda'),
    'action': torch.zeros(2, 4, 6, device='cuda'),
    'action_is_pad': torch.zeros(2, 4, dtype=torch.bool, device='cuda'),
}
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
losses = []
for _ in range(3):
    policy.train()
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    optimizer.step()
    losses.append(float(loss))
policy.reset()
action = policy.select_action(batch)
assert action.shape == (2, 6) and torch.isfinite(action).all()
video = Path('/workspace/course/logs/synthetic-video.mp4')
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=10:duration=1', '-c:v', 'libx264', str(video)], check=True)
import av
with av.open(str(video)) as container:
    av_frames = sum(1 for _ in container.decode(video=0))
from torchcodec.decoders import VideoDecoder
decoder = VideoDecoder(str(video), device='cpu')
assert len(decoder) == av_frames == 10
assert decoder[0].shape == (3, 64, 64)
result = {'kind': 'synthetic-smoke-not-formal-training', 'gpu': torch.cuda.get_device_name(),
          'devices': torch.cuda.device_count(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'act_parameters': sum(p.numel() for p in policy.parameters()), 'losses': losses,
          'action_shape': list(action.shape), 'av_frames': av_frames, 'torchcodec_frames': len(decoder),
          'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
          'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
          'elapsed_s': time.monotonic()-start}
Path('/workspace/course/logs/act-smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
