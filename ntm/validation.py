"""Central validation for DataConfig / TrainConfig / InferConfig and checkpoint load paths."""
import os
from typing import Any, Dict, Optional

from ntm.config import DataConfig, InferConfig, TrainConfig
from ntm.errors import CheckpointFormatError, ConfigValidationError
from ntm.registry import MODEL_REGISTRY

_ALLOWED_CRITERIA = frozenset({"cross_entropy", "bce_softmax", "bce_sigmoid"})


def _prefix(msg: str) -> str:
    return "[ntm] {}".format(msg)


def validate_model_name(model: Any, *, context: str = "model") -> str:
    if model is None or (isinstance(model, str) and not model.strip()):
        raise ConfigValidationError(_prefix("{} must be a non-empty string.".format(context)))
    name = model.strip().lower() if isinstance(model, str) else str(model).lower()
    if name not in MODEL_REGISTRY:
        raise ConfigValidationError(
            _prefix(
                "Unknown {} {!r}; allowed: {}.".format(
                    context, model, ", ".join(sorted(MODEL_REGISTRY.keys()))
                )
            )
        )
    return name


def validate_data_config(cfg: DataConfig) -> None:
    if cfg is None:
        raise ConfigValidationError(_prefix("DataConfig is required."))
    if not isinstance(cfg.taskname, str) or not cfg.taskname.strip():
        raise ConfigValidationError(_prefix("DataConfig.taskname must be a non-empty string."))
    if not isinstance(cfg.lang, str) or not cfg.lang.strip():
        raise ConfigValidationError(_prefix("DataConfig.lang must be a non-empty string."))
    if not isinstance(cfg.no_below, int) or cfg.no_below < 1:
        raise ConfigValidationError(_prefix("DataConfig.no_below must be an integer >= 1."))
    if not _valid_no_above(cfg.no_above):
        raise ConfigValidationError(
            _prefix(
                "DataConfig.no_above must be a fraction in (0, 1) or an absolute doc count >= 1 "
                "(gensim Dictionary.filter_extremes)."
            )
        )


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg is None:
        raise ConfigValidationError(_prefix("TrainConfig is required."))
    validate_model_name(cfg.model, context="TrainConfig.model")

    if not isinstance(cfg.n_topic, int) or cfg.n_topic < 1:
        raise ConfigValidationError(_prefix("TrainConfig.n_topic must be an integer >= 1."))
    if not isinstance(cfg.num_epochs, int) or cfg.num_epochs < 1:
        raise ConfigValidationError(_prefix("TrainConfig.num_epochs must be an integer >= 1."))
    if not isinstance(cfg.batch_size, int) or cfg.batch_size < 1:
        raise ConfigValidationError(_prefix("TrainConfig.batch_size must be an integer >= 1."))
    if not isinstance(cfg.log_every, int) or cfg.log_every < 1:
        raise ConfigValidationError(_prefix("TrainConfig.log_every must be an integer >= 1."))
    if not isinstance(cfg.learning_rate, (int, float)) or cfg.learning_rate <= 0:
        raise ConfigValidationError(_prefix("TrainConfig.learning_rate must be > 0."))
    if not isinstance(cfg.emb_dim, int) or cfg.emb_dim < 1:
        raise ConfigValidationError(_prefix("TrainConfig.emb_dim must be an integer >= 1."))
    if not isinstance(cfg.hid_dim, int) or cfg.hid_dim < 1:
        raise ConfigValidationError(_prefix("TrainConfig.hid_dim must be an integer >= 1."))
    if not isinstance(cfg.dropout, (int, float)) or not (0.0 <= float(cfg.dropout) <= 1.0):
        raise ConfigValidationError(_prefix("TrainConfig.dropout must be in [0, 1]."))

    _validate_device_string(cfg.device, "TrainConfig.device")

    if cfg.criterion not in _ALLOWED_CRITERIA:
        raise ConfigValidationError(
            _prefix(
                "TrainConfig.criterion must be one of {}; got {!r}.".format(
                    ", ".join(sorted(_ALLOWED_CRITERIA)), cfg.criterion
                )
            )
        )

    mn = cfg.model.strip().lower()
    if not isinstance(cfg.dist, str) or not cfg.dist.strip():
        raise ConfigValidationError(_prefix("TrainConfig.dist must be a non-empty string."))

    if cfg.ckpt_path is not None:
        if not isinstance(cfg.ckpt_path, str) or not cfg.ckpt_path.strip():
            raise ConfigValidationError(_prefix("TrainConfig.ckpt_path must be a non-empty path when set."))
        _validate_existing_file(cfg.ckpt_path, "TrainConfig.ckpt_path")


def validate_infer_config(cfg: InferConfig) -> None:
    if cfg is None:
        raise ConfigValidationError(_prefix("InferConfig is required."))
    _validate_device_string(cfg.device, "InferConfig.device")
    if not isinstance(cfg.topk, int) or cfg.topk < 1:
        raise ConfigValidationError(_prefix("InferConfig.topk must be an integer >= 1."))
    if cfg.lang is not None and (not isinstance(cfg.lang, str) or not cfg.lang.strip()):
        raise ConfigValidationError(_prefix("InferConfig.lang must be non-empty when set."))


def validate_checkpoint_path(path: Any, *, label: str = "checkpoint") -> str:
    if path is None or (isinstance(path, str) and not path.strip()):
        raise ConfigValidationError(_prefix("{} path is empty.".format(label)))
    p = path.strip() if isinstance(path, str) else str(path)
    if not os.path.isfile(p):
        raise ConfigValidationError(
            _prefix("{} is not an existing file: {!r}.".format(label, p))
        )
    return p


def validate_load_model_args(
    ckpt_path: Any,
    model_name: Optional[str],
    device: Any,
    taskname: Optional[str],
) -> None:
    validate_checkpoint_path(ckpt_path, label="load_model checkpoint")
    if model_name is not None:
        validate_model_name(model_name, context="load_model model_name")
    _validate_device_string(device, "load_model device")
    if taskname is not None and (not isinstance(taskname, str) or not taskname.strip()):
        raise ConfigValidationError(_prefix("load_model taskname must be non-empty when set."))


def validate_checkpoint_params(model_name: str, params: Dict[str, Any]) -> None:
    """Ensure ``param`` dict from a checkpoint can rebuild the model (inference / load_model)."""
    mn = validate_model_name(model_name, context="checkpoint model_name")
    if not isinstance(params, dict):
        raise ConfigValidationError(_prefix("Checkpoint param must be a dict."))

    for key in ("bow_dim", "n_topic"):
        if key not in params:
            raise ConfigValidationError(
                _prefix('Checkpoint param missing required key {!r} (needed to rebuild model).'.format(key))
            )
    if not isinstance(params["bow_dim"], int) or params["bow_dim"] < 1:
        raise ConfigValidationError(_prefix("Checkpoint param bow_dim must be an integer >= 1."))
    if not isinstance(params["n_topic"], int) or params["n_topic"] < 1:
        raise ConfigValidationError(_prefix("Checkpoint param n_topic must be an integer >= 1."))

    if params.get("taskname") is not None:
        t = params["taskname"]
        if not isinstance(t, str) or not t.strip():
            raise ConfigValidationError(_prefix("Checkpoint param taskname must be non-empty when present."))

    if mn == "wtm":
        for key in ("dist", "dropout"):
            if key not in params:
                raise ConfigValidationError(
                    _prefix("WTM checkpoint param missing {!r} (use a checkpoint saved by ntm or WTM_run).".format(key))
                )
    elif mn == "etm":
        if "emb_dim" not in params:
            raise ConfigValidationError(
                _prefix("ETM checkpoint param missing 'emb_dim' (use a checkpoint saved by ntm or ETM_run).")
            )
    elif mn == "gmntm":
        if "dropout" not in params:
            raise ConfigValidationError(
                _prefix("GMNTM checkpoint param missing 'dropout' (use a checkpoint saved by ntm or GMNTM_run).")
            )
    elif mn == "batm":
        if "hid_dim" not in params:
            raise ConfigValidationError(
                _prefix("BATM checkpoint param missing 'hid_dim' (use a checkpoint saved by ntm or BATM_run).")
            )


def validate_parsed_checkpoint_for_load(parsed: Dict[str, Any], model_name: str) -> None:
    """Reject legacy raw checkpoints and ensure ``param`` can rebuild ``model_name``."""
    fmt = parsed.get("format")
    params = parsed.get("params") or {}
    if fmt == "legacy_state_dict_only":
        raise CheckpointFormatError(
            _prefix(
                "load_model: checkpoint looks like a raw state_dict. "
                "Expected a dict with 'param' and 'net' (from ntm TopicModel.save or *_run.py)."
            )
        )
    if fmt == "legacy_batm_submodules" and not params:
        raise CheckpointFormatError(
            _prefix(
                "load_model: BATM checkpoint has generator/encoder/discriminator tensors but no 'param' block; "
                "re-save with ntm or BATM_run so bow_dim, n_topic, and hid_dim are stored."
            )
        )
    validate_checkpoint_params(model_name, params)


def reformat_checkpoint_error(exc: Exception, *, operation: str) -> CheckpointFormatError:
    """Wrap parse/load failures with a consistent prefix."""
    msg = _prefix("{} failed: {}".format(operation, exc))
    return CheckpointFormatError(msg)


def _validate_existing_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise ConfigValidationError(_prefix("{} is not an existing file: {!r}.".format(label, path)))


def _valid_no_above(val: Any) -> bool:
    """Match gensim: fraction if 0 < val < 1; otherwise absolute count (>= 1)."""
    if val is None or isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    x = float(val)
    if x <= 0:
        return False
    if x < 1.0:
        return True
    return x >= 1.0


def _validate_device_string(device: Any, label: str) -> None:
    if device is None or (isinstance(device, str) and not device.strip()):
        raise ConfigValidationError(_prefix("{} must be non-empty (e.g. 'auto', 'cpu', 'cuda').".format(label)))
    s = device.strip().lower() if isinstance(device, str) else str(device).lower()
    if s == "auto" or s == "cpu":
        return
    if s == "cuda" or s.startswith("cuda:"):
        if s.startswith("cuda:"):
            rest = s.split(":", 1)[1]
            if not rest.isdigit():
                raise ConfigValidationError(_prefix("{}: invalid CUDA device {!r}.".format(label, device)))
        return
    raise ConfigValidationError(
        _prefix("{}: unsupported value {!r}; use 'auto', 'cpu', 'cuda', or 'cuda:N'.".format(label, device))
    )
