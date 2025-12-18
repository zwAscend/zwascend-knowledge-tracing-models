import json
import numpy as np
import mindspore as ms
from mindspore import Tensor

from model import DKTGRU, ModelConfig


def encode_events(events, skill_to_idx, num_skills, max_seq_len=100):
    """
    events: list of dicts: {"skill_id": "...", "correct": 0/1}
    returns x_tokens [1, T] int32
    """
    # take last max_seq_len events (we predict after each, but final mastery uses last step)
    events = events[-max_seq_len:]
    skill_idxs = []
    corrects = []
    for e in events:
        if e["skill_id"] not in skill_to_idx:
            continue
        skill_idxs.append(skill_to_idx[e["skill_id"]])
        corrects.append(int(e["correct"]))

    if len(skill_idxs) == 0:
        raise ValueError("No valid events after filtering unknown skills.")

    x_tokens = np.array(skill_idxs, dtype=np.int32) + np.array(corrects, dtype=np.int32) * num_skills
    x_tokens = x_tokens.reshape(1, -1)  # [B=1, T]
    return x_tokens


def main():
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")


    data_dir = "data/processed"
    ckpt_path = "outputs/checkpoints/best.ckpt"

    with open(f"{data_dir}/meta.json", "r") as f:
        meta = json.load(f)
    K = int(meta["num_skills"])
    skill_to_idx = meta["skill_to_idx"]
    idx_to_skill = {int(k): v for k, v in meta["idx_to_skill"].items()}

    model = DKTGRU(ModelConfig(num_skills=K))
    ms.load_checkpoint(ckpt_path, net=model)
    model.set_train(False)

    # Example events (replace with DB fetch + incoming events)
    events = [
        {"skill_id": "SKILL_LINEAR_EQ_1", "correct": 1},
        {"skill_id": "SKILL_LINEAR_EQ_1", "correct": 0},
        {"skill_id": "SKILL_PERCENT_1", "correct": 1},
    ]

    x_tokens = encode_events(events, skill_to_idx, K, max_seq_len=100)
    p = model(Tensor(x_tokens, ms.int32))  # [1, T, K]

    # take last timestep as "current mastery estimate"
    p_last = p.asnumpy()[0, -1, :]  # [K]

    # return a small map (e.g., top weak skills)
    weakest = np.argsort(p_last)[:10]
    updated_mastery = {idx_to_skill[i]: float(p_last[i]) for i in weakest}

    print("Weakest skills:", updated_mastery)


if __name__ == "__main__":
    main()