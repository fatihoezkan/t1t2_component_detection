"""The matching loss — grading a set of guesses against a set of answers.

The problem it solves: the model emits n_queries compartment guesses in no particular order,
and a voxel has some smaller number of real compartments, also unordered. We can't compare
query 1 to compartment 1 — there's no reason they correspond. So before computing any error
we have to *pair them up*: decide which guess is trying to explain which real compartment.
That's a classic assignment problem, and the Hungarian algorithm solves it optimally (cheapest
total cost). This pairing is the whole idea behind DETR, borrowed here for relaxometry.

Once the pairing is fixed, the rest is ordinary regression on the matched pairs, plus a
separate existence classification that teaches the leftover queries to say "nothing here."

Shapes:
    y_pred : (B, n_queries, 4)   -> [T1, T2, weight, existence_logit] per query
    y_true : (B, max_comp * 3)   -> [T1, T2, weight] per compartment, flattened
    n_comp : (B,)                -> how many of those compartments are real (rest is padding)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Relative importance of each term. With normalized targets these all sit near 1.0.
        self.t1_w = cfg.t1_weight
        self.t2_w = cfg.t2_weight
        self.wt_w = cfg.w_weight
        self.ex_w = cfg.exist_weight
        self.t1_t2_weighting = getattr(cfg, "t1_t2_weighting", "legacy")
        if self.t1_t2_weighting not in {"legacy", "signal_fraction", "uniform"}:
            raise ValueError(
                "t1_t2_weighting must be legacy|signal_fraction|uniform; "
                f"got {self.t1_t2_weighting!r}"
            )

    def forward(self, y_pred, y_true, n_comp):
        device = y_pred.device
        B, n_queries, _ = y_pred.shape
        n_reg = y_pred.shape[-1] - 1                       # 3 regression targets (drop existence)
        y_true = y_true.reshape(B, y_true.shape[1] // n_reg, n_reg)   # (B, max_comp, 3)
        max_comp = y_true.shape[1]

        # --- Step 1: all-pairs cost of assigning every query to every real compartment ---
        # Each of these is (B, n_queries, max_comp): entry [b, q, c] is how badly query q fits
        # compartment c on that term. Broadcasting does the all-pairs work.
        p_t1, p_t2, p_wt = y_pred[:, :, 0:1], y_pred[:, :, 1:2], y_pred[:, :, 2:3]   # (B, Q, 1)
        t_t1 = y_true[:, :, 0].unsqueeze(1)               # (B, 1, C)
        t_t2 = y_true[:, :, 1].unsqueeze(1)
        t_wt = y_true[:, :, 2].unsqueeze(1)

        t1_sq = (p_t1 - t_t1) ** 2
        t2_sq = (p_t2 - t_t2) ** 2
        wt_sq = (p_wt - t_wt) ** 2

        cost = self.t1_w * t1_sq + self.t2_w * t2_sq
        # In both fraction-weighted modes, a dominant pool matters more to the assignment than a
        # weak pool. `uniform` exists as an explicit ablation; it is not used by the new run.
        if self.t1_t2_weighting != "uniform":
            cost = cost * t_wt
        cost = cost + self.wt_w * wt_sq
        # A confident query (high existence) is cheaper to assign, so real compartments
        # preferentially grab queries that already think they exist.
        exist_prob = torch.sigmoid(y_pred[:, :, 3])
        cost = cost + self.ex_w * (1.0 - exist_prob).unsqueeze(2).expand(-1, -1, max_comp)

        # --- Step 2: run the Hungarian algorithm, per voxel ---
        # scipy is CPU/numpy, so we hop off the GPU just for the assignment. This is the one
        # genuinely serial part (one solve per voxel) and why big batches feel heavy on CPU.
        cost_np = cost.detach().cpu().numpy()
        pred_idx, true_idx, batch_idx = [], [], []
        for b in range(B):
            nc = int(n_comp[b].item() if torch.is_tensor(n_comp[b]) else n_comp[b])
            # Match only against the first nc columns — the rest are zero padding. This is
            # exactly why the dataset can zero-fill absent compartments safely.
            rows, cols = linear_sum_assignment(cost_np[b, :, :nc])
            pred_idx.extend(rows)
            true_idx.extend(cols)
            batch_idx.extend([b] * len(rows))

        p = torch.tensor(pred_idx, device=device, dtype=torch.long)
        t = torch.tensor(true_idx, device=device, dtype=torch.long)
        bidx = torch.tensor(batch_idx, device=device, dtype=torch.long)

        # --- Step 3: regression loss on the matched pairs only ---
        matched_w = y_true[bidx, t, 2]                    # true weight of each matched pair
        fraction = matched_w if self.t1_t2_weighting != "uniform" else torch.ones_like(matched_w)
        weighted_t1 = t1_sq[bidx, p, t] * fraction * self.t1_w
        weighted_t2 = t2_sq[bidx, p, t] * fraction * self.t2_w
        weighted_wt = wt_sq[bidx, p, t] * self.wt_w

        # --- Step 4: existence classification for *all* queries ---
        # Label 1 for matched queries, 0 for the rest. Matches are rare (1-3 compartments vs
        # ~10 queries), so up-weight the positives to stop the classifier trivially predicting
        # "nothing everywhere."
        exist_tgt = torch.zeros(B, n_queries, device=device)
        exist_tgt[bidx, p] = 1.0
        pos = exist_tgt.sum(dim=-1)
        pos_weight = torch.clamp((n_queries - pos) / pos.clamp(min=1.0), min=0.5, max=10.0)
        cls_per = F.binary_cross_entropy_with_logits(
            y_pred[:, :, 3], exist_tgt, pos_weight=pos_weight.unsqueeze(1), reduction="none"
        ).mean(dim=-1)                                    # one BCE value per voxel

        # --- Step 5: average matched regression costs back per voxel ---
        # A voxel contributes several matched pairs; index_add sums them per voxel and we divide
        # by the count to get a mean (vectorised — no Python loop over the batch).
        def _per_voxel_mean(vals):
            s = torch.zeros(B, device=device).index_add_(0, bidx, vals)
            c = torch.zeros(B, device=device).index_add_(0, bidx, torch.ones_like(vals))
            return s / c.clamp(min=1.0)

        def _per_voxel_fraction_mean(vals):
            """sum(w * error) / sum(w), not sum(w * error) / n_comp."""
            s = torch.zeros(B, device=device).index_add_(0, bidx, vals)
            w = torch.zeros(B, device=device).index_add_(0, bidx, matched_w)
            return s / w.clamp(min=1e-12)

        if self.t1_t2_weighting == "signal_fraction":
            bt1 = _per_voxel_fraction_mean(weighted_t1)
            bt2 = _per_voxel_fraction_mean(weighted_t2)
        else:
            # `legacy` is the completed baseline's exact reduction; `uniform` is an ordinary
            # compartment mean. Keeping legacy explicit makes old checkpoints/configs repeatable.
            bt1 = _per_voxel_mean(weighted_t1)
            bt2 = _per_voxel_mean(weighted_t2)
        bwt = _per_voxel_mean(weighted_wt)

        # Per-voxel total, then averaged over the batch. Individual terms are returned too so
        # the training log can show where the error is concentrated.
        per_voxel = bt1 + bt2 + bwt + self.ex_w * cls_per
        loss = per_voxel.mean()
        return loss, bt1.mean(), bt2.mean(), bwt.mean(), self.ex_w * cls_per.mean()
