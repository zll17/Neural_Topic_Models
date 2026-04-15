"""Unit tests for ntm.adapters.checkpoint_adapter.parse_checkpoint."""
import pytest
import torch

from ntm.errors import CheckpointFormatError
from ntm.adapters.checkpoint_adapter import parse_checkpoint


def test_parse_legacy_with_net_and_param():
    ckpt = {
        "net": {"w": torch.zeros(1)},
        "param": {"bow_dim": 5, "n_topic": 2},
    }
    out = parse_checkpoint(ckpt)
    assert out["format"] == "legacy_with_meta"
    assert out["params"]["bow_dim"] == 5


def test_parse_v1_format():
    ckpt = {
        "model_state": {"w": torch.zeros(1)},
        "train_config": {"bow_dim": 3, "n_topic": 2},
    }
    out = parse_checkpoint(ckpt)
    assert out["format"] == "v1"


def test_parse_not_dict_raises():
    with pytest.raises(CheckpointFormatError, match="not a dict"):
        parse_checkpoint("x")


def test_parse_unknown_format_raises():
    with pytest.raises(CheckpointFormatError, match="Unknown"):
        parse_checkpoint({"foo": 1, "bar": 2})
