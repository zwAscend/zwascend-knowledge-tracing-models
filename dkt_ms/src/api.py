import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import mindspore as ms
from mindspore import Tensor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import DKTGRU, ModelConfig


# ----------------------------
# Config
# ----------------------------
@dataclass
class AppConfig:
    meta_path: str = os.getenv("DKT_META_PATH", "data/processed/meta.json")
    ckpt_path: str = os.getenv("DKT_CKPT_PATH", "outputs/checkpoints/best.ckpt")
    device: str = os.getenv("DKT_DEVICE", "CPU")
    graph_mode: bool = os.getenv("DKT_GRAPH_MODE", "1") == "1"


# ----------------------------
# Request/Response Schemas
# ----------------------------
class Event(BaseModel):
    skill_id: str
    correct: int = Field(..., ge=0, le=1)
    timestamp: Optional[str] = None  # optional, only for ordering if you pass unsorted events


class MasteryRequest(BaseModel):
    student_id: str
    events: List[Event]
    max_seq_len: int = 100
    return_top_k: int = 10
    return_full_mastery: bool = True


class SkillScore(BaseModel):
    skill_id: str
    mastery: float


class MasteryResponse(BaseModel):
    student_id: str
    num_skills: int
    used_events: int
    ignored_events: int
    top_weak_skills: List[SkillScore]
    mastery: Optional[Dict[str, float]] = None


# ----------------------------
# Model Server
# ----------------------------
class DKTService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        if cfg.graph_mode:
            ms.set_context(mode=ms.GRAPH_MODE)
        else:
            ms.set_context(mode=ms.PYNATIVE_MODE)

        # MindSpore is deprecating device_target in set_context, but this still works.
        # If your version supports ms.set_device("CPU"), you can use that instead.
        try:
            ms.set_device(cfg.device)
        except Exception:
            ms.set_context(device_target=cfg.device)

        # Load meta
        if not os.path.exists(cfg.meta_path):
            raise FileNotFoundError(f"meta.json not found: {cfg.meta_path}")
        meta = json.load(open(cfg.meta_path, "r"))
        self.K = int(meta["num_skills"])
        self.skill_to_idx: Dict[str, int] = meta["skill_to_idx"]
        self.idx_to_skill: Dict[int, str] = {int(k): v for k, v in meta["idx_to_skill"].items()}

        # Load model
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {cfg.ckpt_path}")
        self.model = DKTGRU(ModelConfig(num_skills=self.K))
        ms.load_checkpoint(cfg.ckpt_path, net=self.model)
        self.model.set_train(False)

        # Warm-up compile (helps latency in GRAPH_MODE)
        self._warmup()

    def _warmup(self):
        dummy = Tensor(np.zeros((1, 5), dtype=np.int32), ms.int32)
        _ = self.model(dummy)

    def encode_events(self, events: List[Event], max_seq_len: int):
        # If timestamps exist and caller may send unsorted, you can sort here.
        # For now, we assume caller sends in chronological order.
        events = events[-max_seq_len:]

        skill_idxs = []
        corrects = []
        ignored = 0

        for e in events:
            if e.skill_id not in self.skill_to_idx:
                ignored += 1
                continue
            skill_idxs.append(self.skill_to_idx[e.skill_id])
            corrects.append(int(e.correct))

        if len(skill_idxs) == 0:
            raise ValueError("No usable events after filtering unknown skills.")

        x_skill = np.array(skill_idxs, dtype=np.int32)
        x_corr = np.array(corrects, dtype=np.int32)
        x_tokens = x_skill + x_corr * self.K
        return x_tokens.reshape(1, -1), len(skill_idxs), ignored

    def predict_mastery(self, req: MasteryRequest) -> MasteryResponse:
        x_tokens, used, ignored = self.encode_events(req.events, req.max_seq_len)

        p = self.model(Tensor(x_tokens, ms.int32))  # [1, T, K]
        p_last = p.asnumpy()[0, -1, :]              # [K]

        # Weak skills = lowest mastery
        top_k = max(1, min(req.return_top_k, self.K))
        weakest_idx = np.argsort(p_last)[:top_k]
        top_weak = [
            SkillScore(skill_id=self.idx_to_skill[i], mastery=float(p_last[i]))
            for i in weakest_idx
        ]

        mastery_map = None
        if req.return_full_mastery:
            mastery_map = {self.idx_to_skill[i]: float(p_last[i]) for i in range(self.K)}

        return MasteryResponse(
            student_id=req.student_id,
            num_skills=self.K,
            used_events=used,
            ignored_events=ignored,
            top_weak_skills=top_weak,
            mastery=mastery_map
        )


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="DKT Mastery Service", version="1.0.0")

CFG = AppConfig()
SERVICE: Optional[DKTService] = None


@app.on_event("startup")
def startup():
    global SERVICE
    SERVICE = DKTService(CFG)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/dkt/mastery", response_model=MasteryResponse)
def mastery(req: MasteryRequest):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return SERVICE.predict_mastery(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
