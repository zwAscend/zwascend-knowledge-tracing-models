import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    raw_csv_path: str = "data/raw/interactions.csv"
    out_dir: str = "data/processed"
    max_seq_len: int = 100            # model sequence length (T)
    stride: int = 100                 # non-overlapping windows; try 50 for overlap
    min_events_per_student: int = 2   # must have at least 2 to form (x -> next y)
    seed: int = 42
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train/val/test by student


def _make_windows(skill_idxs: np.ndarray,
                  corrects: np.ndarray,
                  max_seq_len: int,
                  stride: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build windows for DKT:
      - x_tokens[t] encodes (skill_t, correct_t)
      - y_skill[t]  is skill_{t+1}
      - y_correct[t] is correct_{t+1}
      - mask[t] indicates valid (non-pad)
    Each window contains up to (max_seq_len + 1) events to create max_seq_len training steps.
    """
    assert skill_idxs.shape == corrects.shape
    L = len(skill_idxs)
    out = []
    # need at least 2 events to have 1 training step
    if L < 2:
        return out

    for start in range(0, L - 1, stride):
        end = min(start + (max_seq_len + 1), L)
        window_skills = skill_idxs[start:end]
        window_corrects = corrects[start:end]
        if len(window_skills) < 2:
            continue

        # x uses all but last event
        x_skill = window_skills[:-1]
        x_corr = window_corrects[:-1]

        # targets are next-step events
        y_skill = window_skills[1:]
        y_corr = window_corrects[1:]

        T = len(y_skill)  # number of training steps in this window (<= max_seq_len)

        # pad to max_seq_len
        pad_len = max_seq_len - T
        mask = np.ones((T,), dtype=np.float32)

        if pad_len > 0:
            y_skill = np.concatenate([y_skill, np.zeros((pad_len,), dtype=np.int32)])
            y_corr = np.concatenate([y_corr, np.zeros((pad_len,), dtype=np.float32)])
            mask = np.concatenate([mask, np.zeros((pad_len,), dtype=np.float32)])

            x_skill = np.concatenate([x_skill, np.zeros((pad_len,), dtype=np.int32)])
            x_corr = np.concatenate([x_corr, np.zeros((pad_len,), dtype=np.int32)])

        out.append((x_skill.astype(np.int32), x_corr.astype(np.int32),
                    y_skill.astype(np.int32), y_corr.astype(np.float32),
                    mask.astype(np.float32)))
    return out


def main():
    cfg = PreprocessConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.raw_csv_path)
    required = {"student_id", "timestamp", "skill_id", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # normalize types
    df["correct"] = df["correct"].astype(int).clip(0, 1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # map skill_id -> index
    skill_list = sorted(df["skill_id"].astype(str).unique().tolist())
    skill_to_idx: Dict[str, int] = {s: i for i, s in enumerate(skill_list)}
    num_skills = len(skill_list)

    # sort and group by student
    df = df.sort_values(["student_id", "timestamp"])
    students = df["student_id"].astype(str).unique().tolist()

    # split by student (robust for small n)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(students)
    n = len(students)

    # ensure at least 1 student in val and test (when possible)
    n_val = max(1, int(round(cfg.split[1] * n)))
    n_test = max(1, int(round(cfg.split[2] * n)))

    # leave the rest for train
    n_train = n - (n_val + n_test)

    # if dataset is too small, adjust safely
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - (n_train + n_val)  # could be 0 if n==2, but then you need more students

    train_students = set(students[:n_train])
    val_students   = set(students[n_train:n_train + n_val])
    test_students  = set(students[n_train + n_val:n_train + n_val + n_test])


    def build_split(student_set: set):
        X_tokens = []
        Y_skill = []
        Y_corr = []
        Mask = []
        for sid, g in df[df["student_id"].astype(str).isin(student_set)].groupby("student_id"):
            if len(g) < cfg.min_events_per_student:
                continue
            skills = g["skill_id"].astype(str).map(skill_to_idx).to_numpy(dtype=np.int32)
            corr = g["correct"].to_numpy(dtype=np.int32)

            windows = _make_windows(skills, corr, cfg.max_seq_len, cfg.stride)
            for x_skill, x_corr, y_skill, y_corr, mask in windows:
                # encode x_tokens = skill + correct*num_skills => [0..2K-1]
                x_tokens = x_skill + x_corr * num_skills
                X_tokens.append(x_tokens.astype(np.int32))
                Y_skill.append(y_skill.astype(np.int32))
                Y_corr.append(y_corr.astype(np.float32))
                Mask.append(mask.astype(np.float32))

        if not X_tokens:
            raise RuntimeError("No training samples produced. Check CSV size and min_events_per_student.")
        return (np.stack(X_tokens), np.stack(Y_skill), np.stack(Y_corr), np.stack(Mask))

    train = build_split(train_students)
    val = build_split(val_students)
    test = build_split(test_students)

    # save arrays
    np.savez_compressed(os.path.join(cfg.out_dir, "train.npz"),
                        x_tokens=train[0], y_skill=train[1], y_corr=train[2], mask=train[3])
    np.savez_compressed(os.path.join(cfg.out_dir, "val.npz"),
                        x_tokens=val[0], y_skill=val[1], y_corr=val[2], mask=val[3])
    np.savez_compressed(os.path.join(cfg.out_dir, "test.npz"),
                        x_tokens=test[0], y_skill=test[1], y_corr=test[2], mask=test[3])

    meta = {
        "num_skills": num_skills,
        "skill_to_idx": skill_to_idx,
        "idx_to_skill": {str(v): k for k, v in skill_to_idx.items()},
        "max_seq_len": cfg.max_seq_len,
        "stride": cfg.stride
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print("num_skills:", num_skills)
    print("train samples:", train[0].shape[0], "val:", val[0].shape[0], "test:", test[0].shape[0])


if __name__ == "__main__":
    main()
