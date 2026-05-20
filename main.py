"""
MCTL — Multimodal Cognitive Tool-Chain Learning for Fake News Detection

Usage:
    python main.py --exp exp1                          # Exp1: +CAMR
    python main.py --exp exp2                          # Exp2: +CAMR +MCP
    python main.py --exp exp3                          # Exp3: +CAMR +MCP +ATR
    python main.py --exp exp4                          # Exp4: +CAMR +MCP +ATR +MPRE
    python main.py --exp exp5                          # Exp5: Full MCTL (+CBDF)
    python main.py --dataset xfacta --exp exp5         # Run on a specific dataset

Environment variables (see .env.example):
    MCTL_API_KEY, MCTL_API_URL, MCTL_API_MODEL, MCTL_DATA_ROOT
"""
import argparse
import os

# Load .env file if present (requires python-dotenv; silently ignored otherwise)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import (
    CAMRDetector,
    MCPDetector,
    ATRDetector,
    MPREDetector,
    MCTLDetector,
)


def main():
    parser = argparse.ArgumentParser(
        description="MCTL Fake News Detection Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["polifact", "gossip", "weibo21", "weibo", "xfacta"],
        help="Dataset to evaluate on. Omit to run all datasets.",
    )
    parser.add_argument(
        "--exp", type=str, default="exp5",
        choices=["exp1", "exp2", "exp3", "exp4", "exp5"],
        help="Experiment variant to run (default: exp5 = full MCTL).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of retrieved similar samples for CAMR (default: 5).",
    )
    parser.add_argument(
        "--clip_model", type=str, default="ViT-B/32",
        help="CLIP model variant used by CAMR (default: ViT-B/32).",
    )
    parser.add_argument(
        "--text_weight", type=float, default=0.6,
        help="Text embedding weight in CAMR fusion (default: 0.6; "
             "0.8 recommended for Weibo).",
    )
    args = parser.parse_args()

    detectors = {
        "exp1": lambda: CAMRDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp2": lambda: MCPDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp3": lambda: ATRDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp4": lambda: MPREDetector(top_k=args.top_k, clip_model=args.clip_model),
        "exp5": lambda: MCTLDetector(top_k=args.top_k, clip_model=args.clip_model),
    }

    detector = detectors[args.exp]()
    detector.run(dataset_type=args.dataset)


if __name__ == "__main__":
    main()
