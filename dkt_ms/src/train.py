import json
import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.dataset as ds

from model import DKTGRU, ModelConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    out_dir: str = "outputs"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device_target: str = "CPU"


def load_npz(path: str):
    z = np.load(path)
    return z["x_tokens"], z["y_skill"], z["y_corr"], z["mask"]


def make_dataset(x_tokens, y_skill, y_corr, mask, batch_size: int, shuffle: bool):
    def gen():
        for i in range(x_tokens.shape[0]):
            yield (x_tokens[i], y_skill[i], y_corr[i], mask[i])

    dataset = ds.GeneratorDataset(
        gen,
        column_names=["x_tokens", "y_skill", "y_corr", "mask"],
        shuffle=shuffle
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


class DKTLoss(nn.Cell):
    """
    Computes BCE only on the next-skill target:
      p: [B, T, K]
      y_skill: [B, T] (which skill to score at each step)
      y_corr: [B, T] 0/1
      mask: [B, T] 1 valid, 0 padded
    """
    def __init__(self, K: int, eps: float = 1e-7):
        super().__init__()
        self.K = K
        self.eps = eps
        self.onehot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.log = ops.Log()

    def construct(self, p, y_skill, y_corr, mask):
        # y_skill -> onehot [B, T, K]
        on = Tensor(1.0, ms.float32)
        off = Tensor(0.0, ms.float32)
        y_oh = self.onehot(y_skill, self.K, on, off)

        # pick p for the target skill via dot product
        p_next = self.reduce_sum(p * y_oh, -1)  # [B, T]

        # BCE
        p_clamped = ops.clip_by_value(p_next, self.eps, 1.0 - self.eps)
        y = y_corr
        bce = -(y * self.log(p_clamped) + (1.0 - y) * self.log(1.0 - p_clamped))  # [B, T]

        # mask and normalize
        bce = bce * mask
        denom = self.reduce_sum(mask) + self.eps
        return self.reduce_sum(bce) / denom, p_next


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pure-numpy ROC AUC via rank statistic.
    y_true in {0,1}, y_score float.
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Handle ties: average ranks for equal scores
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j+1]].mean()
            ranks[order[i:j+1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def eval_epoch(model, loss_fn, dataset) -> Tuple[float, float]:
    model.set_train(False)
    losses = []
    all_probs = []
    all_labels = []

    for batch in dataset.create_dict_iterator():
        x_tokens = Tensor(batch["x_tokens"], ms.int32)
        y_skill = Tensor(batch["y_skill"], ms.int32)
        y_corr  = Tensor(batch["y_corr"], ms.float32)
        mask    = Tensor(batch["mask"], ms.float32)

        p = model(x_tokens)
        loss, p_next = loss_fn(p, y_skill, y_corr, mask)

        losses.append(float(loss.asnumpy()))

        # collect masked points for AUC
        m = batch["mask"].reshape(-1).astype(np.float32)
        probs = p_next.asnumpy().reshape(-1)
        labels = batch["y_corr"].reshape(-1).astype(np.float32)

        keep = m > 0.5
        all_probs.append(probs[keep])
        all_labels.append(labels[keep])

    mean_loss = float(np.mean(losses))
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs)
    return mean_loss, auc


def main():
    cfg = TrainConfig()
    ms.set_seed(cfg.seed)
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target=cfg.device_target)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")


    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # load meta
    with open(os.path.join(cfg.data_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    K = int(meta["num_skills"])

    # load data
    xtr, ytr_s, ytr_c, mtr = load_npz(os.path.join(cfg.data_dir, "train.npz"))
    xva, yva_s, yva_c, mva = load_npz(os.path.join(cfg.data_dir, "val.npz"))

    train_ds = make_dataset(xtr, ytr_s, ytr_c, mtr, cfg.batch_size, shuffle=True)
    val_ds   = make_dataset(xva, yva_s, yva_c, mva, cfg.batch_size, shuffle=False)

    model = DKTGRU(ModelConfig(num_skills=K))
    loss_fn = DKTLoss(K)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    def forward_fn(x_tokens, y_skill, y_corr, mask):
        p = model(x_tokens)
        loss, _ = loss_fn(p, y_skill, y_corr, mask)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    best_auc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.set_train(True)
        batch_losses = []
        for batch in train_ds.create_dict_iterator():
            x_tokens = Tensor(batch["x_tokens"], ms.int32)
            y_skill = Tensor(batch["y_skill"], ms.int32)
            y_corr  = Tensor(batch["y_corr"], ms.float32)
            mask    = Tensor(batch["mask"], ms.float32)

            loss, grads = grad_fn(x_tokens, y_skill, y_corr, mask)
            optimizer(grads)
            batch_losses.append(float(loss.asnumpy()))

        train_loss = float(np.mean(batch_losses))
        val_loss, val_auc = eval_epoch(model, loss_fn, val_ds)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            print(f"  saved best checkpoint: {ckpt_path} (AUC={best_auc:.4f})")

    # save final info
    with open(os.path.join(cfg.out_dir, "train_summary.json"), "w") as f:
        json.dump({"best_val_auc": best_auc, "num_skills": K}, f, indent=2)


if __name__ == "__main__":
    main()
