from dataclasses import dataclass

import mindspore as ms
from mindspore import nn, ops, Tensor


@dataclass
class ModelConfig:
    num_skills: int
    emb_dim: int = 64
    hidden: int = 128
    dropout: float = 0.0


class DKTGRU(nn.Cell):
    """
    Classic DKT:
      x_tokens: [B, T] values in [0, 2K)
      outputs p: [B, T, K] probability of correctness for each skill at each step
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.K = cfg.num_skills
        self.hidden = cfg.hidden

        self.embed = nn.Embedding(2 * self.K, cfg.emb_dim)
        self.gru = nn.GRU(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden,
            num_layers=1,
            has_bias=True,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=False
        )
        self.head = nn.Dense(cfg.hidden, self.K)
        self.sigmoid = ops.Sigmoid()

    def construct(self, x_tokens):
        # x_tokens: [B, T] int32
        x = self.embed(x_tokens)  # [B, T, emb_dim]
        B = x.shape[0]
        h0 = ops.zeros((1, B, self.hidden), ms.float32)
        out, _ = self.gru(x, h0)  # out: [B, T, hidden]
        logits = self.head(out)   # [B, T, K]
        p = self.sigmoid(logits)
        return p
