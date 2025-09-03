# `ntm` 库使用说明（API 与示例）

本仓库将神经主题模型封装为可导入的 **`ntm`** 包（与 PyPI **发行名** `neural-topic-models` 对应；代码中仍为 `import ntm`）。原有 `*_run.py` 命令行脚本保留不变。

---

## 安装

包元数据在 **`pyproject.toml`**（PEP 621）与 **`setup.cfg`** 中各有一份，请保持版本与依赖一致；构建后端为 **setuptools**（`setuptools>=45`），兼容常见镜像上的版本。

### 从源码安装（推荐开发时使用）

在克隆后的仓库根目录执行：

```shell
pip install -e .
```

这会安装声明的核心依赖（PyTorch、gensim、jieba 等）。若需与旧版 `requirements.txt` 更接近（含可选组件），可额外：

```shell
pip install -e ".[full]"
```

若可编辑安装失败（例如旧版 `pip` 提示需要 setuptools 构建），可先升级打包工具后再试：`python -m pip install -U pip setuptools wheel`。

或继续单独安装 `requirements.txt` 中的条目（如 HanLP、spaCy 模型等）。

### 仅依赖脚本、不使用 `ntm` API

仍可使用：

```shell
pip install -r requirements.txt
```

---

## 公开 API 一览

| 符号 | 说明 |
|------|------|
| `train_model` | 便捷训练：构建数据与模型、训练、保存 checkpoint |
| `load_model` | 从 `{"param","net"}` 格式 checkpoint 加载 |
| `infer_topics` | 对文本列表做主题推断（需词典可用时） |
| `TopicModel` | 训练/保存/推理的面向对象入口 |
| `DataConfig`, `TrainConfig`, `InferConfig` | 数据与训练、推理配置 |
| `TrainResult`, `TopicPrediction` | 训练与推理结果类型 |
| `ConfigValidationError`, `CheckpointFormatError` | 参数与 checkpoint 相关错误 |
| `validate_data_config`, `validate_train_config`, `validate_infer_config`, `validate_checkpoint_params` | 可选的显式校验（入口已自动调用部分校验） |

**支持的模型名**（`train_model` 的 `model=` / `TrainConfig.model`）：`gsm`、`wtm`、`etm`、`gmntm`、`batm`（大小写不敏感）。

---

## 快速示例：训练 → 加载 → 推理

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
preds = infer_topics(
    loaded,
    ["china economy growth policy market"],
    topk=2,
)
print(result.ckpt_path, preds[0].top_topics)
```

### `train_model` 常用参数

| 参数 | 含义 |
|------|------|
| `model` | 模型名，见上表 |
| `taskname` | 数据子目录名，对应 `data/<taskname>/` 与语料约定 |
| `n_topic` | 主题数 |
| `num_epochs`, `batch_size`, `log_every` | 训练轮数、批大小、日志间隔 |
| `device` | `"auto"` / `"cpu"` / `"cuda"` / `"cuda:N"` |
| `lang`, `no_below`, `no_above`, `rebuild`, `no_rebuild`, `auto_adj`, `use_tfidf` | 与 `DataConfig` 一致 |
| `criterion`, `dist`, `emb_dim`, `dropout`, `hid_dim`, `ckpt` / `ckpt_path` | 与 `TrainConfig` 一致；WTM/GMNTM 等默认与库内约定一致 |
| `save_dir` | 保存 checkpoint 的目录，默认 `ckpt` |

返回值 **`TrainResult`** 字段：`model_name`, `taskname`, `n_topic`, `ckpt_path`, `metrics`（BATM 可能为空字典）。

### `load_model`

```python
model = load_model(path, model_name="wtm", device="auto", taskname=None)
```

- 历史 checkpoint 建议显式传入 `model_name`。
- 若 checkpoint 为 ntm v1 且含 `model_name`，可省略（见错误提示）。

### `infer_topics` / `TopicPrediction`

- `infer_topics(topic_model, texts, topk=3)` 返回 `TopicPrediction` 列表。
- 每项含 `text`, `topic_distribution`, `top_topics`。
- 推理默认按**空格**切分；中文建议传入已分词、空格连接的字符串。

---

## 进阶：`TopicModel` + 配置类

需要与 CLI 对齐或自定义数据管线时，可直接使用配置对象：

```python
from ntm import TopicModel, DataConfig, TrainConfig

data_cfg = DataConfig(
    taskname="cnews10k",
    lang="zh",
    no_below=5,
    no_above=0.005,
)
train_cfg = TrainConfig(
    model="wtm",
    n_topic=10,
    num_epochs=2,
    batch_size=2048,
    device="auto",
)

topic_model, train_data = TopicModel.from_config(train_cfg, data_cfg)
result = topic_model.fit(train_data, train_cfg)
topic_model.save("my_run.ckpt")
```

- **`DataConfig`**：`taskname`, `lang`, `no_below`, `no_above`, `rebuild`, `no_rebuild`, `auto_adj`, `use_tfidf`。
- **`TrainConfig`**：`model`, `n_topic`, `num_epochs`, `batch_size`, `criterion`, `device`, `ckpt_path`, `log_every`, `learning_rate`, `dist`, `emb_dim`, `dropout`, `hid_dim`。
- **`TopicModel.infer(texts, infer_cfg=None)`**：可传入 `InferConfig(device=..., topk=..., lang=...)`。

---

## 校验与错误

入口会对 `DataConfig` / `TrainConfig` / `InferConfig` 做校验；也可在自定义脚本中提前调用：

```python
from ntm import DataConfig, TrainConfig, validate_data_config, validate_train_config

validate_data_config(DataConfig(taskname="demo"))
validate_train_config(TrainConfig(model="wtm"))
```

- **`ConfigValidationError`**：参数不合法（如 `n_topic < 1`、未知模型名）。
- **`CheckpointFormatError`**：文件无法解析为支持的 checkpoint 结构。

---

## 与旧脚本的关系

- 训练产物的 checkpoint 布局与 `*_run.py` 一致：`{"param": ..., "net": ...}`，可供 `inference.py` 与 `load_model` 共用。
- 完整依赖与可选 NLP 组件仍以仓库内 `requirements.txt` 为准；`pyproject.toml` 提供可安装的**最小核心依赖**与 **`[full]`** 可选扩展。

---

## 开发与测试

在仓库根目录安装开发依赖后运行：

```shell
pip install -e ".[dev]"
pytest
```

`pytest` 配置在 `pyproject.toml` 的 `[tool.pytest.ini_options]`（`pythonpath` 包含仓库根目录，以便导入 `models` 与 `ntm`）。测试位于 `tests/`：校验逻辑单元测试、`parse_checkpoint` 单元测试、以及最小 **`load_model` 集成测试**（构造小型 GSM checkpoint）。

---

## 版本说明

当前库版本与 `pyproject.toml` 中 `[project] version` 一致；发布到 PyPI 时使用发行名 **`neural-topic-models`**，导入名仍为 **`ntm`**。
