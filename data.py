"""data.py  –  Data loading utilities for the FL‑IDS project.

Key features
============
* **Single load / global split:** the full UNSW‑NB15 CSV corpus is read **once**
  at import time, then split 80 / 20 into a *train pool* and a *global
  hold‑out* test set.
* **Memory‑friendly preprocessing:** only numeric columns (≈ 40 features) are
  used; NaNs are median‑imputed and then standard‑scaled.  Missing
  `attack_cat` labels are mapped to the string "normal" so benign traffic is
  preserved.
* **Dirichlet label‑skew partitioner:** a single α hyper‑parameter controls how
  non‑IID each client shard is.  α→∞ ⇒ IID ; α→0.1 ⇒ highly skewed.
* **Optional visualisation:** helper to plot class distributions per client so
  you can *see* the skew.

This file is *import‑side‑effect free* unless executed directly (`python
data.py`), in which case it prints dataset sizes and can optionally verify
shard/test disjointness.
"""

from __future__ import annotations

import os
import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 0.  Globals / user‑configurable constants
# ---------------------------------------------------------------------------

SEED: int = 42
DATA_DIR: Path = Path("dataset/10%")                 # root with UNSW‑NB15 CSVs
BATCH_SIZE: int = 256
ALPHA_DIRICHLET: float = 999999                    # lower = more skew

# rng used **everywhere** to keep determinism
RNG: np.random.Generator = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1.  CSV loading helper (recursive)
# ---------------------------------------------------------------------------

def _load_csv_folder(folder: Path) -> pd.DataFrame:
    """Read every *.csv under *folder* (non‑recursively) into one DataFrame."""
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {folder!s}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------------
# 2.  Pre‑processing – numeric‑only pipeline
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (float32), y (int64) after basic cleaning.

    Steps
    -----
    1. Replace ±∞ → NaN.
    2. Fill missing/empty labels with "normal".
    3. Keep only numeric columns (≈ 40 features in UNSW‑NB15).
    4. Median‑impute NaNs and standard‑scale.
    5. Label‑encode `attack_cat`.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df["attack_cat"] = df["attack_cat"].fillna("normal").replace("", "normal")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Impute → scale
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    scaler = StandardScaler()
    X = scaler.fit_transform(df[num_cols]).astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df["attack_cat"].astype(str)).astype(np.int64)
    print("le classes " , le.classes_)
    return X, y, le.classes_

# ---------------------------------------------------------------------------
# 3.  Load full dataset exactly once, then split 80/20
# ---------------------------------------------------------------------------

def _load_and_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_raw = _load_csv_folder(DATA_DIR)
    print(f"Raw rows before preprocess: {len(df_raw):,}")
    X_full, y_full, class_names = _preprocess(df_raw)
    print(f"Rows after preprocess:        {len(X_full):,} (features={X_full.shape[1]})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=SEED,
    )
    return X_train, y_train, X_test, y_test, class_names

# materialise at import time
X_TRAIN_POOL, Y_TRAIN_POOL, X_TEST_GLOBAL, Y_TEST_GLOBAL, CLASS_NAMES = _load_and_split()
print(
    f"Global split     ➜ train={len(X_TRAIN_POOL):,}  test={len(X_TEST_GLOBAL):,}"
)
# Print counts per class
classes, train_counts = np.unique(Y_TRAIN_POOL, return_counts=True)
test_counts = np.array([ (Y_TEST_GLOBAL==c).sum() for c in classes ])
print("\nData counts per class:")
for c, tc, vc in zip(classes, train_counts, test_counts):
    name = CLASS_NAMES[c] if CLASS_NAMES is not None else str(c)
    print(f" Class {c} ({name}): train={tc}, test={vc}")



# ---------------------------------------------------------------------------
# 4.  Dirichlet label‑skew partitioner
# ---------------------------------------------------------------------------

def _dirichlet_label_split(
    y: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Return list of index arrays, one per client, with Dirichlet label skew."""
    cls_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(y):
        cls_indices[int(label)].append(idx)
     #   if idx % 50000 == 0:
      #      print(f"At idx={idx}, label={label}, cls_indices sizes: {[len(v) for v in cls_indices.values()]}")
    #print("cls_indices sizes:" ,{len(cls_indices[9])}) 
   # print("cls " , cls_indices[9])
    client_shards: list[list[int]] = [[] for _ in range(num_clients)]
    for label, idxs in cls_indices.items():
    #    print("label ", label, " idx ", idxs)
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
     #   print("proportions is" , proportions)
        split_pts = (np.cumsum(proportions) * len(idxs)).astype(int)
       # print("split " , split_pts)
        shards = np.split(idxs, split_pts[:-1])
        for cid, shard in enumerate(shards):
            client_shards[cid].extend(shard)

    return [np.array(s, dtype=np.int32) for s in client_shards]

# ---------------------------------------------------------------------------
# 5.  Public Loader API used by server & clients
# ---------------------------------------------------------------------------

def load_global_test(batch_size: int = BATCH_SIZE) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_TEST_GLOBAL),
        torch.from_numpy(Y_TEST_GLOBAL),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_partition(
    cid: int,
    num_clients: int,
    batch_size: int = BATCH_SIZE,
    alpha: float = ALPHA_DIRICHLET,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for client *cid* under Dirichlet skew."""




    # Build shards once per call – cheap, uses only label vector
    client_indices = _dirichlet_label_split(Y_TRAIN_POOL, num_clients, alpha, RNG)
    part = client_indices[cid]
    plot_label_distribution(client_indices, Y_TRAIN_POOL)
    plot_class_counts(Y_TRAIN_POOL, Y_TEST_GLOBAL, class_names=None)
    X_part = X_TRAIN_POOL[part]
    y_part = Y_TRAIN_POOL[part]

    # Count number of samples per class
    unique, counts = np.unique(y_part, return_counts=True)

    # Safe stratify only if all classes have at least 2 samples
    if np.any(counts < 2) or len(unique) < 2:
        stratify = None
        print(f"[Client {cid}] Stratify disabled (too few samples per class)")
    else:
        stratify = y_part


    # 80/20 validation inside the shard
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_part,
        y_part,
        test_size=0.2,
        stratify=stratify,
        random_state=SEED,
    )

    print(
        f"[Client {cid}] α={alpha}  » train {len(X_tr):,}  val {len(X_val):,}  classes {len(np.unique(y_tr))}"
    )

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

# ---------------------------------------------------------------------------
# 6.  Optional visualisation utility
# ---------------------------------------------------------------------------

def plot_label_distribution(client_indices: List[np.ndarray], y: np.ndarray):
    """Stacked‑bar chart of label counts per client (requires matplotlib)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    class_ids = np.unique(y)
    data = {
        f"Client {cid}": np.bincount(y[shard], minlength=len(class_ids))
        for cid, shard in enumerate(client_indices)
    }
    df = pd.DataFrame(data, index=[f"C{c}" for c in class_ids])
    df.T.plot(kind="bar", stacked=True, figsize=(10, 4))
    plt.ylabel("sample count")
    plt.title(
        f"Dirichlet label skew (α = {ALPHA_DIRICHLET}) – class distribution per client"
    )
    plt.tight_layout()
    plt.show()


def plot_class_counts(y_train: np.ndarray, y_test: np.ndarray, class_names: list[str] | None = None):
    """
    Bar‐chart of number of samples per class in train vs test.
    y_train, y_test: 1D arrays of integer labels
    class_names: optional list mapping label→name
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute counts
    classes = np.unique(np.concatenate([y_train, y_test]))
    train_counts = [ (y_train == c).sum() for c in classes ]
    test_counts  = [ (y_test  == c).sum() for c in classes ]

    labels = class_names if class_names is not None else [str(c) for c in classes]
    x = np.arange(len(classes))

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, train_counts, width, label="Train")
    ax.bar(x + width/2, test_counts,  width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("Class distribution: train vs. test")
    ax.legend()
    plt.tight_layout()
    plt.show()




# ---------------------------------------------------------------------------
# 7.  Quick self‑test / diagnostic mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_CLIENTS = 4
    print("\n=== Diagnostics ===")
    # Build shards once for diagnostics (does not affect loaders)
    shards = _dirichlet_label_split(Y_TRAIN_POOL, NUM_CLIENTS, ALPHA_DIRICHLET, RNG)
    print("shards " , shards)
    # Check disjointness with global test
    test_set = set(map(tuple, np.hstack([X_TEST_GLOBAL, Y_TEST_GLOBAL[:, None]])))
    for cid, shard in enumerate(shards):
        overlap = 0
        for idx in shard:
            row = tuple(np.hstack([X_TRAIN_POOL[idx], [Y_TRAIN_POOL[idx]]]))
            if row in test_set:
                overlap += 1
        print(f"Client {cid}: overlap with global test = {overlap}")

    # Optional: visualise label distribution
    try:
        plot_label_distribution(shards, Y_TRAIN_POOL)
    except ImportError:
        print("matplotlib not installed – skipping plot")

