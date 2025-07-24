from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TopicPrediction:
    text: str
    topic_distribution: List[float]
    top_topics: List[int]


@dataclass
class TrainResult:
    model_name: str
    taskname: str
    n_topic: int
    ckpt_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
