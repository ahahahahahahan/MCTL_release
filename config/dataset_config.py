"""
Dataset configuration.

DATA_ROOT can be set via the MCTL_DATA_ROOT environment variable.
Default: ./data  (relative to the project root)

Expected directory layout:
  data/
  ├── polifact/
  │   ├── train.jsonl
  │   ├── test.jsonl
  │   └── images/
  ├── gossip/
  ├── weibo21/
  ├── weibo/
  └── xfacta/

Each .jsonl file should contain one JSON object per line with at least the
fields: "text", "labels" (list), and optionally "image" (filename).
"""
import os

DATA_ROOT = os.environ.get("MCTL_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))

DATASET_CONFIGS = {
    "polifact": {
        "test_path": os.path.join(DATA_ROOT, "polifact", "test.jsonl"),
        "train_path": os.path.join(DATA_ROOT, "polifact", "train.jsonl"),
        "images_dir": os.path.join(DATA_ROOT, "polifact", "images"),
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/polifact",
    },
    "gossip": {
        "test_path": os.path.join(DATA_ROOT, "gossip", "test.jsonl"),
        "train_path": os.path.join(DATA_ROOT, "gossip", "train.jsonl"),
        "images_dir": os.path.join(DATA_ROOT, "gossip", "images"),
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/gossip",
    },
    "weibo21": {
        "test_path": os.path.join(DATA_ROOT, "weibo21", "test.jsonl"),
        "train_path": os.path.join(DATA_ROOT, "weibo21", "train.jsonl"),
        "images_dir": os.path.join(DATA_ROOT, "weibo21", "images"),
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/weibo21",
    },
    "weibo": {
        "test_path": os.path.join(DATA_ROOT, "weibo", "test.jsonl"),
        "train_path": os.path.join(DATA_ROOT, "weibo", "train.jsonl"),
        "images_dir": os.path.join(DATA_ROOT, "weibo", "images"),
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/weibo",
    },
    "xfacta": {
        "test_path": os.path.join(DATA_ROOT, "xfacta", "test.jsonl"),
        "train_path": os.path.join(DATA_ROOT, "xfacta", "train.jsonl"),
        "images_dir": os.path.join(DATA_ROOT, "xfacta", "images"),
        "batch_size": 30,
        "max_concurrency": 15,
        "results_dir": "results/xfacta",
    },
}
