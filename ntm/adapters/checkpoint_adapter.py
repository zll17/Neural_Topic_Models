import torch

from ntm.errors import CheckpointFormatError


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)


def parse_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        raise CheckpointFormatError("Checkpoint is not a dict.")

    if "net" in ckpt:
        return {
            "format": "legacy_with_meta",
            "model_state": ckpt["net"],
            "optimizer_state": ckpt.get("optimizer"),
            "epoch": ckpt.get("epoch"),
            "params": ckpt.get("param", {}),
        }

    if "model_state" in ckpt:
        return {
            "format": "v1",
            "model_state": ckpt["model_state"],
            "optimizer_state": ckpt.get("optimizer_state"),
            "epoch": ckpt.get("epoch"),
            "params": ckpt.get("train_config", {}),
        }

    # Historical format: directly saved state_dict
    if _looks_like_state_dict(ckpt):
        return {
            "format": "legacy_state_dict_only",
            "model_state": ckpt,
            "optimizer_state": None,
            "epoch": None,
            "params": {},
        }

    # BATM historical format: three submodules only
    if {"generator", "encoder", "discriminator"}.issubset(ckpt.keys()):
        return {
            "format": "legacy_batm_submodules",
            "model_state": ckpt,
            "optimizer_state": None,
            "epoch": None,
            "params": {},
        }

    raise CheckpointFormatError("Unknown checkpoint format.")


def _looks_like_state_dict(payload):
    if not payload:
        return False
    sample_value = next(iter(payload.values()))
    return torch.is_tensor(sample_value)
