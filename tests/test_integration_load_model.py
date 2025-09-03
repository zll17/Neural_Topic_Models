"""Integration test: build a tiny GSM checkpoint and load via ntm.load_model."""
import torch

from models.GSM import GSM

from ntm.facade import load_model


def test_load_model_gsm_roundtrip(tmp_path):
    bow_dim = 80
    n_topic = 3
    taskname = "pytest_task"
    device = torch.device("cpu")
    gsm = GSM(bow_dim=bow_dim, n_topic=n_topic, taskname=taskname, device=device)
    ckpt_dict = {
        "net": gsm.vae.state_dict(),
        "param": {
            "bow_dim": bow_dim,
            "n_topic": n_topic,
            "taskname": taskname,
        },
    }
    path = tmp_path / "gsm_smoke.ckpt"
    torch.save(ckpt_dict, path)

    loaded = load_model(str(path), model_name="gsm", device="cpu")
    assert loaded.model_name == "gsm"
    assert loaded.n_topic == n_topic
    assert loaded.taskname == taskname
