from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    taskname: str
    lang: str = "zh"
    no_below: int = 5
    no_above: float = 0.005
    rebuild: bool = False
    # If True: same as CLI --no_rebuild (use cached corpus when present; rebuild_eff = not no_rebuild)
    no_rebuild: bool = False
    auto_adj: bool = False
    use_tfidf: bool = False


@dataclass
class TrainConfig:
    model: str
    n_topic: int = 20
    num_epochs: int = 100
    batch_size: int = 512
    criterion: str = "cross_entropy"
    device: str = "auto"
    ckpt_path: Optional[str] = None
    log_every: int = 10
    learning_rate: float = 1e-3
    dist: str = "gmm_std"
    emb_dim: int = 300
    dropout: float = 0.0
    hid_dim: int = 1024  # BATM GAN hidden width (matches models/BATM default)


@dataclass
class InferConfig:
    device: str = "auto"
    lang: Optional[str] = None
    topk: int = 3
