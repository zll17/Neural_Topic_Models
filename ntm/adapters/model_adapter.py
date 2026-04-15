import torch

from ntm.errors import UnsupportedModelError
from ntm.registry import get_model_class


def resolve_device(device_value):
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def build_model(train_cfg, bow_dim, taskname):
    model_name = train_cfg.model.lower()
    device = resolve_device(train_cfg.device)
    model_cls = get_model_class(model_name)

    kwargs = {
        "bow_dim": bow_dim,
        "n_topic": train_cfg.n_topic,
        "taskname": taskname,
        "device": device,
    }
    if model_name == "wtm":
        kwargs.update({"dist": train_cfg.dist, "dropout": train_cfg.dropout})
    elif model_name == "etm":
        kwargs.update({"emb_dim": train_cfg.emb_dim})
    elif model_name == "gmntm":
        kwargs.update({"dropout": train_cfg.dropout})
    elif model_name == "batm":
        kwargs.update({"hid_dim": train_cfg.hid_dim})

    return model_cls(**kwargs), device


def build_model_from_params(model_name, params, device):
    model_cls = get_model_class(model_name)
    init_params = _filter_init_params(model_name, dict(params))
    init_params["device"] = resolve_device(device)
    return model_cls(**init_params)


def load_model_state(model_name, model_obj, parsed_ckpt):
    name = model_name.lower()
    fmt = parsed_ckpt["format"]
    state = parsed_ckpt["model_state"]

    if name == "batm":
        if isinstance(state, dict) and "generator" in state:
            model_obj.generator.load_state_dict(state["generator"])
            model_obj.encoder.load_state_dict(state["encoder"])
            model_obj.discriminator.load_state_dict(state["discriminator"])
            return
        raise UnsupportedModelError("BATM checkpoint 'net' must contain generator, encoder, discriminator.")

    model_obj.load_model(state)


def _filter_init_params(model_name, params):
    common = {"bow_dim", "n_topic", "taskname"}
    name = model_name.lower()
    if name == "wtm":
        allow = common.union({"dist", "dropout"})
    elif name == "etm":
        allow = common.union({"emb_dim"})
    elif name == "gmntm":
        allow = common.union({"dropout"})
    elif name == "batm":
        allow = common.union({"hid_dim"})
    else:
        allow = common

    filtered = {}
    for key, value in params.items():
        if key in allow and value is not None:
            filtered[key] = value
    return filtered
