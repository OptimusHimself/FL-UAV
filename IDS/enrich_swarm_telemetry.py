"""
Augment data/swarm_processed/*.npz windows with 2 aligned telemetry channels per timestep:
  (battery_soc %, cpu_util %).

Run after preprocess_swarm_federated.py.

- If X has 44 features: appends synthesized battery + CPU (becomes 46).
- If X has 47 features (legacy battery + speed + CPU): drops the speed column (becomes 46).
- If X already has 46 features: skips that file.

Usage:
  python enrich_swarm_telemetry.py
  python enrich_swarm_telemetry.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from telemetry_synth import synthesize_telemetry

SWARM_DIR = Path("data/swarm_processed")
META_PATH = SWARM_DIR / "meta.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    normal_id = int(meta["label_to_id"]["Normal"])

    files = sorted(SWARM_DIR.glob("train_split_*.npz")) + [SWARM_DIR / "test.npz"]
    changed = False

    for path in files:
        if not path.is_file():
            continue
        d = np.load(path, allow_pickle=True)
        if "X_train" in d:
            X = np.asarray(d["X_train"], dtype=np.float32)
            y = np.asarray(d["y_train"], dtype=np.int64)
            xkey = "X_train"
        else:
            X = np.asarray(d["X_test"], dtype=np.float32)
            y = np.asarray(d["y_test"], dtype=np.int64)
            xkey = "X_test"

        if X.ndim != 3 or X.shape[2] not in (44, 46, 47):
            raise ValueError(f"{path}: unexpected X shape {X.shape}")

        if X.shape[2] == 46:
            print(f"skip (already 46-d): {path.name}")
            continue

        if X.shape[2] == 47:
            # legacy layout: ...44 IDS..., battery, gps_speed, cpu
            X_new = np.concatenate([X[:, :, :45], X[:, :, 46:]], axis=2).astype(np.float32)
            bak = path.with_suffix(path.suffix + ".bak47")
        else:
            sub_seed = (args.seed + abs(hash(path.name)) % 10_000_000) % 2**32
            rng = np.random.default_rng(sub_seed)
            extra = synthesize_telemetry(X, y, normal_id, rng)
            X_new = np.concatenate([X, extra], axis=2).astype(np.float32)
            bak = path.with_suffix(path.suffix + ".bak44")

        shutil.copy2(path, bak)
        ykey = "y_train" if xkey == "X_train" else "y_test"
        y_arr = np.asarray(d[ykey], dtype=np.int64)
        if xkey == "X_train":
            np.savez_compressed(path, X_train=X_new, y_train=y_arr)
        else:
            np.savez_compressed(path, X_test=X_new, y_test=y_arr)
        print(f"updated {path.name}: {X.shape} -> {X_new.shape}  (backup {bak.name})")
        changed = True

    meta["n_features"] = 46
    meta["telemetry_channels"] = [
        "battery_soc_percent",
        "cpu_util_percent",
    ]
    meta["telemetry_model"] = (
        "SOC decreases with frame.len proxy + attack IDS uplift; "
        "CPU = baseline + sqrt(load) + burst|Δload| + attack uplift, clipped. "
        "See telemetry_synth.py docstring for rationale."
    )
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if changed:
        print("Updated meta.json n_features=46")
    else:
        print("All npz splits already 46-d; meta.json synced to n_features=46")


if __name__ == "__main__":
    main()
