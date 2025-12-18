import json
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.dataset as ds

from model import DKTGRU, ModelConfig
from train import DKTLoss, make_dataset, load_npz, roc_auc_score


def main():
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")


    data_dir = "data/processed"
    out_dir = "outputs"
    ckpt_path = os.path.join(out_dir, "checkpoints", "best.ckpt")

    with open(os.path.join(data_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    K = int(meta["num_skills"])

    xt, ys, yc, m = load_npz(os.path.join(data_dir, "test.npz"))
    test_ds = make_dataset(xt, ys, yc, m, batch_size=64, shuffle=False)

    model = DKTGRU(ModelConfig(num_skills=K))
    ms.load_checkpoint(ckpt_path, net=model)

    loss_fn = DKTLoss(K)

    model.set_train(False)
    losses = []
    all_probs = []
    all_labels = []

    for batch in test_ds.create_dict_iterator():
        x_tokens = Tensor(batch["x_tokens"], ms.int32)
        y_skill = Tensor(batch["y_skill"], ms.int32)
        y_corr  = Tensor(batch["y_corr"], ms.float32)
        mask    = Tensor(batch["mask"], ms.float32)

        p = model(x_tokens)
        loss, p_next = loss_fn(p, y_skill, y_corr, mask)
        losses.append(float(loss.asnumpy()))

        probs = p_next.asnumpy().reshape(-1)
        labels = batch["y_corr"].reshape(-1).astype(np.float32)
        keep = batch["mask"].reshape(-1) > 0.5
        all_probs.append(probs[keep])
        all_labels.append(labels[keep])

    mean_loss = float(np.mean(losses))
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs)

    print(f"TEST | loss={mean_loss:.4f} | AUC={auc:.4f} | points={len(labels)}")


if __name__ == "__main__":
    main()