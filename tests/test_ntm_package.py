"""Smoke tests for ntm package exports."""
import ntm


def test_ntm_all_exported():
    expected = {
        "CheckpointFormatError",
        "ConfigValidationError",
        "DataConfig",
        "InferConfig",
        "TrainConfig",
        "TopicModel",
        "TopicPrediction",
        "TrainResult",
        "train_model",
        "load_model",
        "infer_topics",
        "validate_checkpoint_params",
        "validate_data_config",
        "validate_infer_config",
        "validate_train_config",
    }
    assert set(ntm.__all__) == expected


def test_train_result_dataclass():
    from ntm import TrainResult

    r = TrainResult(model_name="gsm", taskname="t", n_topic=5)
    assert r.ckpt_path is None
    assert r.metrics == {}
