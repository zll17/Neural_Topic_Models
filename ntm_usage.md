# ntm 最小使用示例

本文档给出 `ntm` 包在当前阶段的最小可运行流程：**训练 -> 加载 -> 推理**。

## 1) 基础导入

```python
from ntm import train_model, load_model, infer_topics
```

## 2) 训练（最小示例）

```python
result = train_model(
    model="wtm",          # 可选: gsm / wtm / etm / gmntm / batm
    taskname="cnews10k",  # 对应 data/cnews10k_lines.txt
    n_topic=5,
    num_epochs=1,         # smoke test 建议 1
    batch_size=4096,      # 可适当调大以加快单 epoch
    log_every=1,
    device="auto",        # 自动选择 cuda/cpu
)
print("checkpoint:", result.ckpt_path)
print("metrics:", result.metrics)
```

返回值 `result` 是 `TrainResult`，包含：

- `model_name`
- `taskname`
- `n_topic`
- `ckpt_path`
- `metrics`

## 3) 加载模型

```python
model = load_model(result.ckpt_path, model_name="wtm")
```

说明：

- 新格式 checkpoint 可自动识别模型名；
- 历史 checkpoint 建议显式传入 `model_name`。

## 4) 推理

```python
preds = infer_topics(
    model,
    [
        "china economy growth policy market",
        "basketball team season player game",
    ],
    topk=2,
)

for p in preds:
    print("text:", p.text)
    print("top_topics:", p.top_topics)
    print("dist_len:", len(p.topic_distribution))
```

`infer_topics` 返回 `TopicPrediction` 列表，每项包含：

- `text`
- `topic_distribution`
- `top_topics`

## 5) 一次性完整示例

```python
from ntm import train_model, load_model, infer_topics

result = train_model(
    model="wtm",
    taskname="cnews10k",
    n_topic=5,
    num_epochs=1,
    batch_size=4096,
    log_every=1,
    device="auto",
)

loaded = load_model(result.ckpt_path, model_name="wtm")
preds = infer_topics(loaded, ["china economy growth policy market"], topk=2)

print("ckpt:", result.ckpt_path)
print("top topics:", preds[0].top_topics)
print("distribution size:", len(preds[0].topic_distribution))
```

## 6) 当前阶段注意事项

- 当前推理默认将输入文本按空格切分（`text.split()`），建议输入已分词文本（以空格分隔）。
- 训练时依赖项目现有数据与分词链路，建议先确保对应 `taskname` 数据可用。
- 本文档用于 Phase 1 验证；后续会在 Phase 2 增加更完整的 tokenizer 和推理接口。
