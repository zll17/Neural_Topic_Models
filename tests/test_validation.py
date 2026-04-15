"""Unit tests for ntm.validation."""
import pytest

from ntm.config import DataConfig, InferConfig, TrainConfig
from ntm.errors import CheckpointFormatError, ConfigValidationError
from ntm.validation import (
    reformat_checkpoint_error,
    validate_checkpoint_params,
    validate_data_config,
    validate_infer_config,
    validate_model_name,
    validate_parsed_checkpoint_for_load,
    validate_train_config,
)


def test_validate_model_name_ok():
    assert validate_model_name("GSM") == "gsm"


def test_validate_model_name_unknown():
    with pytest.raises(ConfigValidationError, match="Unknown"):
        validate_model_name("not_a_model")


def test_validate_data_config_ok():
    validate_data_config(DataConfig(taskname="t", lang="zh", no_below=3, no_above=0.1))
    validate_data_config(DataConfig(taskname="t", no_above=100))


def test_validate_data_config_bad_taskname():
    with pytest.raises(ConfigValidationError, match="taskname"):
        validate_data_config(DataConfig(taskname="  "))


def test_validate_train_config_ok():
    validate_train_config(TrainConfig(model="wtm", dist="gmm_std"))


def test_validate_train_config_bad_n_topic():
    with pytest.raises(ConfigValidationError, match="n_topic"):
        validate_train_config(TrainConfig(model="gsm", n_topic=0))


def test_validate_train_config_ckpt_path_must_exist(tmp_path):
    p = tmp_path / "resume.pt"
    p.write_bytes(b"x")
    cfg = TrainConfig(model="wtm", dist="gmm_std", ckpt_path=str(p))
    validate_train_config(cfg)


def test_validate_train_config_ckpt_missing_file(tmp_path):
    missing = tmp_path / "nope.pt"
    cfg = TrainConfig(model="wtm", dist="gmm_std", ckpt_path=str(missing))
    with pytest.raises(ConfigValidationError, match="not an existing file"):
        validate_train_config(cfg)


def test_validate_infer_config_ok():
    validate_infer_config(InferConfig(device="cpu", topk=2))


def test_validate_checkpoint_params_gsm():
    validate_checkpoint_params(
        "gsm",
        {"bow_dim": 10, "n_topic": 2, "taskname": "demo"},
    )


def test_validate_checkpoint_params_wtm_requires_dist_dropout():
    with pytest.raises(ConfigValidationError, match="dist"):
        validate_checkpoint_params("wtm", {"bow_dim": 10, "n_topic": 2})


def test_validate_parsed_checkpoint_rejects_raw_state_dict():
    parsed = {
        "format": "legacy_state_dict_only",
        "params": {},
        "model_state": {},
    }
    with pytest.raises(CheckpointFormatError, match="raw state_dict"):
        validate_parsed_checkpoint_for_load(parsed, "gsm")


def test_reformat_checkpoint_error_wraps_message():
    err = reformat_checkpoint_error(ValueError("bad"), operation="op")
    assert isinstance(err, CheckpointFormatError)
    assert "[ntm]" in str(err)
    assert "op" in str(err)
