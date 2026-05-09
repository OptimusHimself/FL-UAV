"""
Synthetic UAV-side telemetry aligned 1:1 with existing (N, T, F) IDS windows.

Design rationale (simplified, citation-friendly):
- **Battery (SOC %)**: Coulomb-style discrete model — SOC decreases in proportion to an
  electrical *load proxy* derived from per-packet `frame.len` (traffic intensity) and a
  constant *IDS / radio processing* uplift on attack windows. See energy-aware UAV path
  planning / SOC–load coupling (e.g. linear SOC–energy models in UAV scheduling literature).
- **CPU (%)**: Edge-style utilization — idle baseline, sublinear cost in traffic intensity
  via sqrt(`frame.len` proxy), extra cost on |Δload| bursts (scheduling / cache / queue
  pressure), and a constant uplift on attack windows (deeper inspection path), clipped.

Outputs shape (N, T, 2) in order: [battery_percent, cpu_percent].
"""

from __future__ import annotations

import numpy as np

FRAME_LEN_COL = 1  # index in 44-d Case1-aligned features


def _attack_mask(y: np.ndarray, normal_class_id: int) -> np.ndarray:
    return (y.astype(np.int64) != normal_class_id).astype(np.float32)


def synthesize_telemetry(
    X: np.ndarray,
    y: np.ndarray,
    normal_class_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    X: (N, T, F) standardized IDS features (base IDS dims only, e.g. F=44)
    y: (N,) window-level class ids
    returns Telem: (N, T, 2) — battery %, CPU %
    """
    n, t, _f = X.shape
    load = np.abs(X[:, :, FRAME_LEN_COL].astype(np.float64))
    load = load / (np.median(load) + 1e-3)

    atk = _attack_mask(y, normal_class_id)[:, None]  # (N,1)

    telem = np.zeros((n, t, 2), dtype=np.float32)

    # --- Battery: random initial SOC, then per-step decrement ~ load + attack IDS cost ---
    soc0 = rng.uniform(58.0, 99.0, size=n).astype(np.float32)
    k_load, k_atk, sigma_b = 0.35, 0.55, 0.12
    soc = soc0
    for step in range(t):
        telem[:, step, 0] = soc
        drain = k_load * load[:, step] + k_atk * atk[:, 0] + rng.normal(0.0, sigma_b, size=n)
        drain = np.clip(drain.astype(np.float32), 0.05, 2.5)
        soc = np.clip(soc - drain, 6.0, 100.0)

    # --- CPU: baseline + sqrt(load) work + burst |Δload| + attack inspection uplift ---
    burst = np.zeros((n, t), dtype=np.float64)
    burst[:, 1:] = np.abs(load[:, 1:] - load[:, :-1])
    sqrt_load = np.sqrt(np.maximum(load, 1e-6))
    base_idle, w_sqrt, w_burst, w_atk_cpu, sigma_c = 9.0, 24.0, 11.0, 14.0, 3.5
    for step in range(t):
        cpu = (
            base_idle
            + w_sqrt * sqrt_load[:, step]
            + w_burst * burst[:, step]
            + w_atk_cpu * atk[:, 0]
            + rng.normal(0.0, sigma_c, size=n)
        )
        telem[:, step, 1] = np.clip(cpu.astype(np.float32), 4.0, 99.0)

    return telem
