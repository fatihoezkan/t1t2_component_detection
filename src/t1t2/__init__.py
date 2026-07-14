"""t1t2 — a from-scratch DETR training package for T1–T2 correlation MRI.

The thesis treats microstructure recovery as *set prediction*: a Detection Transformer
reads a voxel's 64-point (8 TI × 8 TE) signal and predicts the set of water compartments
{(T1, T2, weight)} plus an existence score, matched to ground truth by the Hungarian
algorithm so order never matters.

The public surface is deliberately small — import the piece you need:

    from t1t2.config import load_config, ExperimentConfig
    from t1t2.data   import VoxelDataset, make_dataloader, TargetNormalizer
    from t1t2.model  import build_model, T1T2DETR
    from t1t2.loss   import HungarianLoss
"""

__version__ = "0.1.0"
