"""ACT motion loss and a separate stop probability for segmented demonstrations."""

import torch
from lerobot.policies.act.modeling_act import OBS_IMAGES
from torch.nn import functional as F


def segmented_loss(policy, batch, stop_weight):
    model_batch = dict(batch)
    model_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    predicted, (mu, log_variance) = policy.model(model_batch)
    actual = batch["action"]
    valid = ~batch["action_is_pad"]
    motion = ((predicted[..., :10] - actual[..., :10]).abs() * valid.unsqueeze(-1)).mean()
    last = actual[..., 10]
    weight = torch.as_tensor(stop_weight, device=last.device, dtype=last.dtype)
    frame_loss = F.binary_cross_entropy_with_logits(
        predicted[..., 10], last, pos_weight=weight, reduction="none"
    )
    denominator = ((1 + (weight - 1) * last) * valid).sum().clamp_min(1)
    boundary = (frame_loss * valid).sum() / denominator
    kl = (-0.5 * (1 + log_variance - mu.square() - log_variance.exp())).sum(-1).mean()
    return motion + kl * policy.config.kl_weight + 0.2 * boundary, boundary.detach()
