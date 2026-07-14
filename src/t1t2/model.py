"""The detector: a DETR that finds tissue compartments in a voxel's decay curve.

Think object detection, but the "image" is a 64-number signal and the "objects" are water
pools. A fixed set of learned queries each go looking for one compartment; after attending
to the encoded signal through a transformer decoder, every query reports a guess —
(T1, T2, weight) plus a score for "is there really a compartment here?". The clever part,
matching guesses to real compartments without caring about order, lives in the loss
(loss.py), not here. This file is only the network that produces the guesses.

Faithful to the prior diffusion-DETR of Schlund et al. / the correlation-imaging-detr_t1t2
repository the thesis builds on; rewritten clean and driven by a ModelConfig, so an
experiment's YAML fully describes the architecture.
"""
from __future__ import annotations

import torch
from torch import nn


class SignalEncoder(nn.Module):
    """The 64-point signal -> one feature vector, via a 4-layer MLP (LayerNorm + ReLU).

    This is the "backbone": it compresses the whole decay curve into the fs_dim-wide memory
    vector the decoder queries attend to.
    """

    def __init__(self, input_dim: int, hidden_dim: int, fs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fs_dim), nn.LayerNorm(fs_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    """A plain three-layer MLP. Each prediction head (T1, T2, weight, existence) is one of
    these — small, boring, easy to reason about."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class T1T2DETR(nn.Module):
    """Signal in, set of compartments out.

    Flow: encode the 64-point signal into one feature vector, let a fixed number of learned
    queries attend to it through a transformer decoder, then read each query out through the
    heads. Output is (B, n_queries, 4) where the last axis is [T1, T2, weight, existence_logit].
    There are always n_queries guesses; which ones are "real" is decided later by the
    existence score.
    """

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.fs_dim = cfg.fs_dim
        self.n_queries = cfg.n_queries
        self.n_layers = cfg.n_dlayers
        self.n_heads = cfg.n_heads
        self.aux_loss = cfg.aux_loss

        self.encoder = SignalEncoder(self.input_dim, self.hidden_dim, self.fs_dim)
        # Optional warm-start: load encoder weights from a prior checkpoint, and optionally
        # freeze them so the borrowed features stay fixed. Kept minimal — a fuller
        # pretraining story (autoencoder) is a later extension.
        if cfg.pretrain_path:
            state = torch.load(cfg.pretrain_path, weights_only=True, map_location="cpu")
            self.encoder.load_state_dict(state)
            if cfg.freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False

        # The queries are learned vectors — n_queries "slots," each learning to specialise in
        # a kind of compartment.
        self.queries = nn.Embedding(self.n_queries, self.fs_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            self.fs_dim, self.n_heads, dim_feedforward=self.hidden_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.n_layers, norm=nn.LayerNorm(self.fs_dim)
        )

        # One small MLP per regression target.
        self.t1_head = MLPHead(self.fs_dim, self.fs_dim, 1)
        self.t2_head = MLPHead(self.fs_dim, self.fs_dim, 1)
        self.w_head = MLPHead(self.fs_dim, self.fs_dim, 1)

        # Existence is scored differently: instead of judging each query alone, we squash every
        # query down and let one head look at all of them together, so it can reason about the
        # whole set ("I've already claimed two compartments; this third looks redundant").
        # red_dim keeps the concatenated vector from getting huge.
        red_dim = self.fs_dim // 4
        self.qu_red = nn.Sequential(nn.Linear(self.fs_dim, red_dim), nn.LayerNorm(red_dim), nn.ReLU())
        self.exist_head = MLPHead(red_dim * self.n_queries, self.fs_dim, self.n_queries)

    def forward(self, X):
        B = X.size(0)
        memory = self.encoder(X).unsqueeze(1)                     # (B, 1, fs) — what queries attend to
        hs = self.queries.weight.unsqueeze(0).expand(B, -1, -1)   # (B, n_q, fs) — same queries per voxel

        # Step through decoder layers by hand (rather than one self.decoder(...) call) so that,
        # when aux_loss is on, we can read a prediction off every intermediate layer and
        # supervise it too — it gives earlier layers a gradient signal and trains more stably.
        aux = []
        for i in range(self.n_layers):
            hs = self.decoder.layers[i](tgt=hs, memory=memory)
            if self.aux_loss:
                aux.append(self._predict(hs))

        hs = self.decoder.norm(hs)                                # final LayerNorm
        out = self._predict(hs)
        if self.aux_loss:
            return {"pred": out, "aux": aux}
        return out

    def _predict(self, hs):
        """Run the heads on a set of decoder states and assemble the (B, n_q, 4) output."""
        # sigmoid keeps regressed values in [0, 1] — the same range the normalized targets live
        # in, so predictions and targets are directly comparable.
        t1 = torch.sigmoid(self.t1_head(hs))
        t2 = torch.sigmoid(self.t2_head(hs))
        w = torch.sigmoid(self.w_head(hs))
        # existence stays as raw logits; the loss uses BCE-with-logits (numerically stable).
        hs_cat = self.qu_red(hs).reshape(hs.size(0), -1)
        exist = self.exist_head(hs_cat).unsqueeze(2)
        return torch.cat([t1, t2, w, exist], dim=-1)


def build_model(model_cfg) -> T1T2DETR:
    """Tiny factory so callers don't import the class directly — handy if the model grows
    variants selected by config later."""
    return T1T2DETR(model_cfg)
