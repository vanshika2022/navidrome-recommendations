"""
Popularity-based cold start blender — serving side.

Loads pre-computed popularity scores (a .npy file produced by
train/cold_start.py::ColdStartRecommender.save_popularity()) and
blends GRU4Rec scores with popularity scores based on session length.

Blend weight alpha = min(session_len / ramp_sessions, 1.0)
  alpha=0  -> pure popularity   (empty / very short session)
  alpha=1  -> pure GRU4Rec      (session_len >= ramp_sessions)
"""

from __future__ import annotations

import numpy as np
import torch


class ColdStartBlender:
    """
    Parameters
    ----------
    pop_scores    : (num_items,) float32 numpy array of pre-computed
                    log-smoothed popularity scores, 0-indexed (item_idx - 1).
    ramp_sessions : number of known interactions before fully trusting GRU4Rec.
    """

    def __init__(self, pop_scores: np.ndarray, ramp_sessions: int = 3):
        self.ramp_sessions = ramp_sessions
        self._pop = torch.from_numpy(pop_scores.astype(np.float32))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str, ramp_sessions: int = 3) -> ColdStartBlender:
        """Load from a .npy file saved by train/cold_start.py."""
        pop_scores = np.load(path)
        return cls(pop_scores, ramp_sessions=ramp_sessions)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def alpha(self, session_len: int) -> float:
        """Blend coefficient: 0 = pure popularity, 1 = pure GRU4Rec."""
        return min(session_len / self.ramp_sessions, 1.0)

    @torch.no_grad()
    def predict(
        self,
        model,
        prefix_items: torch.Tensor,     # (B, L) right-padded, 0 = pad
        user_idxs: torch.Tensor,        # (B,)
        all_item_emb: torch.Tensor,     # (num_items, D)
        top_n: int,
        exclude_sets: list | None = None,
    ) -> tuple[list[list[int]], list[list[float]], list[float]]:
        """
        Returns (indices, scores, alphas):
            indices : list[B] of list[top_n] 1-based item indices
            scores  : list[B] of list[top_n] blended log-prob scores
            alphas  : list[B] of per-sample blend coefficients (0=pop, 1=gru)
        """
        device = prefix_items.device
        B      = prefix_items.size(0)

        if exclude_sets is None:
            exclude_sets = [set() for _ in range(B)]

        # Session lengths (non-pad tokens per sample)
        lengths = (prefix_items != 0).sum(dim=1)   # (B,)

        # GRU4Rec session scores
        raw_model    = model.module if hasattr(model, "module") else model
        session_repr = raw_model.encode_session(prefix_items, user_idxs)  # (B, D)
        gru_scores   = session_repr @ all_item_emb.T                      # (B, num_items)

        # Align popularity to model vocab size (finetune model may be smaller)
        model_n  = all_item_emb.shape[0]
        pop      = self._pop[:model_n].to(device)                         # (model_n,)
        gru_log  = torch.log_softmax(gru_scores, dim=-1)                  # (B, model_n)
        pop_log  = torch.log_softmax(
            pop.unsqueeze(0).expand(B, -1), dim=-1
        )                                                                  # (B, model_n)

        all_indices, all_scores, all_alphas = [], [], []
        for b in range(B):
            a      = self.alpha(int(lengths[b].item()))
            scores = a * gru_log[b] + (1.0 - a) * pop_log[b]

            for item_idx in exclude_sets[b]:
                arr_idx = item_idx - 1
                if 0 <= arr_idx < scores.size(0):
                    scores[arr_idx] = float("-inf")

            top = torch.topk(scores, top_n)
            all_indices.append([int(i) + 1 for i in top.indices.cpu().tolist()])
            all_scores.append([float(s) for s in top.values.cpu().tolist()])
            all_alphas.append(round(a, 3))

        return all_indices, all_scores, all_alphas
