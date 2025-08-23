import os
import time

import numpy as np
import torch
from gensim.corpora import Dictionary

from ntm.adapters.checkpoint_adapter import load_checkpoint, parse_checkpoint
from ntm.adapters.model_adapter import resolve_device
from ntm.config import DataConfig, InferConfig, TrainConfig
from ntm.errors import UnsupportedModelError
from ntm.types import TopicPrediction, TrainResult


class TopicModel:
    def __init__(
        self,
        model_name,
        model_obj,
        dictionary=None,
        device=None,
        taskname=None,
        n_topic=None,
        save_param_extra=None,
    ):
        self.model_name = model_name.lower()
        self.model = model_obj
        self.dictionary = dictionary
        self.device = device if isinstance(device, torch.device) else resolve_device(device or "cpu")
        self.taskname = taskname
        self.n_topic = n_topic
        self._save_param_extra = dict(save_param_extra or {})

    @classmethod
    def from_config(cls, train_cfg: TrainConfig, data_cfg: DataConfig):
        from ntm.adapters.dataset_adapter import build_train_dataset
        from ntm.adapters.model_adapter import build_model

        train_data = build_train_dataset(data_cfg)
        model_obj, device = build_model(train_cfg, train_data.vocabsize, data_cfg.taskname)
        save_extra = {}
        mn = train_cfg.model.lower()
        if mn == "batm":
            save_extra["hid_dim"] = train_cfg.hid_dim
        if mn == "gmntm":
            save_extra["dropout"] = train_cfg.dropout
        topic_model = cls(
            model_name=train_cfg.model,
            model_obj=model_obj,
            dictionary=train_data.dictionary,
            device=device,
            taskname=data_cfg.taskname,
            n_topic=train_cfg.n_topic,
            save_param_extra=save_extra,
        )
        return topic_model, train_data

    def fit(self, train_data, train_cfg: TrainConfig):
        kwargs = {
            "train_data": train_data,
            "batch_size": train_cfg.batch_size,
            "test_data": train_data,
            "num_epochs": train_cfg.num_epochs,
            "log_every": train_cfg.log_every,
        }

        if self.model_name in ("gsm", "etm"):
            kwargs["criterion"] = train_cfg.criterion
        if self.model_name == "gmntm":
            kwargs["criterion"] = train_cfg.criterion
        if self.model_name == "batm":
            kwargs.pop("test_data")

        if train_cfg.ckpt_path:
            kwargs["ckpt"] = torch.load(
                train_cfg.ckpt_path,
                map_location=self.device,
            )

        self.model.train(**kwargs)
        metrics = self._evaluate(train_data)
        return TrainResult(
            model_name=self.model_name,
            taskname=self.taskname or "",
            n_topic=self.n_topic or 0,
            metrics=metrics,
        )

    def _evaluate(self, test_data):
        if self.model_name == "batm":
            return {}
        c_v, c_w2v, c_uci, c_npmi, mimno_tc, td = self.model.evaluate(test_data=test_data)
        return {
            "c_v": c_v,
            "c_w2v": c_w2v,
            "c_uci": c_uci,
            "c_npmi": c_npmi,
            "mimno_tc": mimno_tc,
            "td": td,
        }

    def infer(self, texts, infer_cfg: InferConfig = None):
        if self.dictionary is None:
            raise ValueError("Dictionary is required for text inference.")
        if infer_cfg is None:
            infer_cfg = InferConfig()

        preds = []
        for text in texts:
            tokens = text.split()
            dist = self.model.inference(doc_tokenized=tokens, dictionary=self.dictionary).tolist()
            top_topics = np.argsort(np.array(dist))[::-1][: infer_cfg.topk].tolist()
            preds.append(
                TopicPrediction(
                    text=text,
                    topic_distribution=dist,
                    top_topics=top_topics,
                )
            )
        return preds

    def save(self, path):
        """Same layout as CLI *_run.py: ``{\"param\": ..., \"net\": ...}`` for inference.py."""
        param = {
            "bow_dim": self.model.bow_dim,
            "n_topic": self.model.n_topic,
            "taskname": self.taskname,
        }
        param.update(self._save_param_extra)
        if self.model_name == "wtm":
            param["dist"] = self.model.dist
            param["dropout"] = self.model.dropout
        elif self.model_name == "etm":
            param["emb_dim"] = self.model.emb_dim
        torch.save(
            {
                "net": _extract_state_dict(self.model_name, self.model),
                "param": param,
            },
            path,
        )

    def get_topic_words(self, topk=15):
        if self.dictionary is None:
            return self.model.show_topic_words(topK=topk)
        return self.model.show_topic_words(topK=topk, dictionary=self.dictionary)


def train_model(model, taskname, n_topic=20, num_epochs=100, **kwargs):
    ml = (model or "").lower()
    default_dropout = 0.0
    if ml == "wtm":
        default_dropout = 0.4
    elif ml == "gmntm":
        default_dropout = 0.2
    default_criterion = "cross_entropy"
    if ml == "gmntm":
        default_criterion = "bce_softmax"

    data_cfg = DataConfig(
        taskname=taskname,
        lang=kwargs.pop("lang", "zh"),
        no_below=kwargs.pop("no_below", 5),
        no_above=kwargs.pop("no_above", 0.005),
        rebuild=kwargs.pop("rebuild", False),
        no_rebuild=kwargs.pop("no_rebuild", False),
        auto_adj=kwargs.pop("auto_adj", False),
        use_tfidf=kwargs.pop("use_tfidf", False),
    )
    train_cfg = TrainConfig(
        model=model,
        n_topic=n_topic,
        num_epochs=num_epochs,
        batch_size=kwargs.pop("batch_size", 512),
        criterion=kwargs.pop("criterion", default_criterion),
        device=kwargs.pop("device", "auto"),
        dist=kwargs.pop("dist", "gmm_std"),
        emb_dim=kwargs.pop("emb_dim", 300),
        dropout=kwargs.pop("dropout", default_dropout),
        hid_dim=kwargs.pop("hid_dim", 1024),
        log_every=kwargs.pop("log_every", 10),
        ckpt_path=kwargs.pop("ckpt", None) or kwargs.pop("ckpt_path", None),
    )

    topic_model, train_data = TopicModel.from_config(train_cfg, data_cfg)
    result = topic_model.fit(train_data, train_cfg)

    save_dir = kwargs.pop("save_dir", "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = "{}_{}_tp{}_{}.ckpt".format(
        model.upper(),
        taskname,
        n_topic,
        time.strftime("%Y-%m-%d-%H-%M", time.localtime()),
    )
    ckpt_path = os.path.join(save_dir, ckpt_name)
    topic_model.save(ckpt_path)
    result.ckpt_path = ckpt_path
    return result


def load_model(ckpt_path, model_name=None, device="auto", taskname=None):
    from ntm.adapters.model_adapter import build_model_from_params, load_model_state

    map_dev = resolve_device(device)
    ckpt = load_checkpoint(ckpt_path, map_location=map_dev)
    parsed = parse_checkpoint(ckpt)

    if model_name is None:
        if isinstance(ckpt, dict) and ckpt.get("format_version") == 1:
            model_name = ckpt.get("model_name")
        else:
            raise ValueError("model_name is required unless checkpoint contains model_name (ntm v1 format).")

    params = dict(parsed["params"])
    if "bow_dim" not in params or "n_topic" not in params:
        raise ValueError("Checkpoint missing required model params: bow_dim and n_topic.")

    model_obj = build_model_from_params(model_name, params, device)
    load_model_state(model_name, model_obj, parsed)

    model_taskname = params.get("taskname", taskname)
    dictionary = _load_dictionary(model_taskname) if model_taskname else None
    save_extra = {}
    mn = (model_name or "").lower()
    if mn == "batm" and "hid_dim" in params:
        save_extra["hid_dim"] = params["hid_dim"]
    if mn == "gmntm" and "dropout" in params:
        save_extra["dropout"] = params["dropout"]
    return TopicModel(
        model_name=model_name,
        model_obj=model_obj,
        dictionary=dictionary,
        device=map_dev,
        taskname=model_taskname,
        n_topic=params.get("n_topic"),
        save_param_extra=save_extra,
    )


def infer_topics(topic_model, texts, topk=3):
    infer_cfg = InferConfig(topk=topk)
    return topic_model.infer(texts, infer_cfg=infer_cfg)


def _load_dictionary(taskname):
    cwd = os.getcwd()
    dict_path = os.path.join(cwd, "data", taskname, "dict.txt")
    if not os.path.exists(dict_path):
        return None
    return Dictionary.load_from_text(dict_path)


def _extract_state_dict(model_name, model_obj):
    name = model_name.lower()
    if name in ("gsm", "etm"):
        return model_obj.vae.state_dict()
    if name == "wtm":
        return model_obj.wae.state_dict()
    if name == "gmntm":
        return model_obj.vade.state_dict()
    if name == "batm":
        return {
            "generator": model_obj.generator.state_dict(),
            "encoder": model_obj.encoder.state_dict(),
            "discriminator": model_obj.discriminator.state_dict(),
        }
    raise UnsupportedModelError("Unsupported model: {}".format(model_name))
