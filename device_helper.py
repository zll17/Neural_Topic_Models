"""Shared helpers for CLI training/inference scripts."""
import torch


def default_device():
    """Use CUDA when available, otherwise CPU (avoids failing on CPU-only machines)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
