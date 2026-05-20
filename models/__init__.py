"""
Models package — one file per MCTL module/ablation variant.

Experiment hierarchy (each row adds one module to the previous):

  camr.py  → CAMRDetector  (Exp1 : + CAMR retrieval)
  mcp.py   → MCPDetector   (Exp2 : + Meta-Cognitive Planning)
  atr.py   → ATRDetector   (Exp3 : + Adaptive Tool Router)
  mpre.py  → MPREDetector  (Exp4 : + Multi-Perspective Reasoning Ensemble)
  mctl.py  → MCTLDetector  (Exp5 : + Consensus-Based Decision Fusion)
"""
from .camr import CAMRDetector
from .mcp import MCPDetector
from .atr import ATRDetector
from .mpre import MPREDetector
from .mctl import MCTLDetector
