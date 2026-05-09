import json
from typing import Any

import flwr as fl
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from model import CnnLstmIDS, set_model_parameters
from scenario_config import format_scenario_block, scenario_for_round
from swarm_fl_data import SWARM_DIR, load_global_test, load_meta


def _fit_config(server_round: int) -> dict[str, float]:
    return {"server_round": float(server_round)}


def _eval_config(server_round: int) -> dict[str, float]:
    return {"server_round": float(server_round)}


X_te, y_te, NUM_CLASSES, N_FEATURES = load_global_test()
X_test = torch.tensor(X_te, dtype=torch.float32)
y_test = torch.tensor(y_te, dtype=torch.long)


def _fmt_pct_ratio(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _fmt_pct_0_100(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}%"


def _telemetry_summary_from_tensor(x: torch.Tensor) -> dict[str, float] | None:
    """Mean battery (%) and CPU (%) over test windows if telemetry channels exist."""
    if x.dim() != 3 or x.size(-1) < 46:
        return None
    bat = float(x[:, :, 44].mean().item())
    cpu = float(x[:, :, 45].mean().item())
    return {"battery_soc_mean": bat, "cpu_util_mean": cpu}


def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        net = CnnLstmIDS(n_features=N_FEATURES, num_classes=NUM_CLASSES)
        set_model_parameters(net, parameters)

        net.eval()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = net(X_test)
            loss = criterion(logits, y_test).item()
            _, predicted = torch.max(logits.data, 1)

        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / total if total else 0.0

        sc = scenario_for_round(server_round)
        telem = _telemetry_summary_from_tensor(X_test)

        payload = {
            "phase": "centralized_eval",
            "server_round": int(server_round),
            "scenario": {
                "region": sc["region"],
                "place": sc["place"],
                "latitude_deg": sc["latitude_deg"],
                "longitude_deg": sc["longitude_deg"],
                "gps_wgs84": sc["gps_wgs84"],
                "training_scene": sc["training_scene"],
            },
            "test": {
                "windows": int(total),
                "path": str(SWARM_DIR / "test.npz"),
                "accuracy": accuracy,
                "loss": loss,
                "precision_macro": float(prec),
                "recall_macro": float(rec),
            },
        }
        if telem:
            payload["test_pooled_telemetry_mean"] = telem
            payload["test_pooled_telemetry_mean_fmt"] = {
                "battery_soc_mean": _fmt_pct_0_100(telem["battery_soc_mean"]),
                "cpu_util_mean": _fmt_pct_0_100(telem["cpu_util_mean"]),
            }
        payload["test_fmt"] = {
            "accuracy": _fmt_pct_ratio(accuracy),
            "precision_macro": _fmt_pct_ratio(float(prec)),
            "recall_macro": _fmt_pct_ratio(float(rec)),
        }

        print("\n" + "=" * 56)
        print(format_scenario_block(sc, title=f"[SERVER] round {server_round} — eval theatre"))
        print("-" * 56)
        print(
            "[SERVER] eval summary: "
            f"acc={_fmt_pct_ratio(accuracy)}  "
            f"precision_macro={_fmt_pct_ratio(float(prec))}  "
            f"recall_macro={_fmt_pct_ratio(float(rec))}  "
            f"loss={loss:.4f}"
        )
        if telem:
            print(
                "[SERVER] pooled telemetry mean: "
                f"battery={_fmt_pct_0_100(telem['battery_soc_mean'])}  "
                f"CPU={_fmt_pct_0_100(telem['cpu_util_mean'])}"
            )
        print("-" * 56)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 56 + "\n")

        metrics_out: dict[str, Any] = {
            "accuracy": accuracy,
            "precision_macro": float(prec),
            "recall_macro": float(rec),
        }
        if telem:
            metrics_out["battery_soc_mean"] = telem["battery_soc_mean"]
            metrics_out["cpu_util_mean"] = telem["cpu_util_mean"]
        return loss, metrics_out

    return evaluate


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with structured console logs after each fit aggregation."""

    def aggregate_fit(self, server_round, results, failures):
        sc = scenario_for_round(server_round)
        rows: list[dict[str, Any]] = []
        for item in results or []:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            cp, fit_res = item
            cid = getattr(cp, "cid", "?")
            metrics = getattr(fit_res, "metrics", None) or {}
            n_ex = int(getattr(fit_res, "num_examples", 0) or 0)
            rows.append(
                {
                    "client_id": cid,
                    "num_examples": n_ex,
                    "battery_soc_mean": metrics.get("battery_soc_mean"),
                    "cpu_util_mean": metrics.get("cpu_util_mean"),
                    "battery_soc_mean_fmt": _fmt_pct_0_100(metrics.get("battery_soc_mean")),
                    "cpu_util_mean_fmt": _fmt_pct_0_100(metrics.get("cpu_util_mean")),
                }
            )

        fit_payload = {
            "phase": "fit_aggregate",
            "server_round": int(server_round),
            "scenario": {
                "region": sc["region"],
                "place": sc["place"],
                "latitude_deg": sc["latitude_deg"],
                "longitude_deg": sc["longitude_deg"],
                "gps_wgs84": sc["gps_wgs84"],
                "training_scene": sc["training_scene"],
            },
            "drones": rows,
            "failures": len(failures or []),
        }

        print("\n" + "=" * 56)
        print(format_scenario_block(sc, title=f"[SERVER] round {server_round} — fit complete"))
        print("-" * 56)
        if rows:
            joined = "  ".join(
                f"UAV-{r['client_id']}: battery={r['battery_soc_mean_fmt']} CPU={r['cpu_util_mean_fmt']}"
                for r in rows
            )
            print(f"[SERVER] client telemetry snapshot: {joined}")
            print("-" * 56)
        print(json.dumps(fit_payload, indent=2, ensure_ascii=False))
        print("=" * 56 + "\n")

        return super().aggregate_fit(server_round, results, failures)


_kw_base = dict(
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=2,
    evaluate_fn=get_evaluate_fn(),
)
try:
    strategy = LoggingFedAvg(
        **_kw_base,
        on_fit_config_fn=_fit_config,
        on_evaluate_config_fn=_eval_config,
    )
except TypeError:
    strategy = LoggingFedAvg(**_kw_base)


if __name__ == "__main__":
    load_meta()
    print("[SERVER] Swarm UAV IDS (CNN+LSTM) + FedAvg")
    print(f"   data: {SWARM_DIR}  X_test shape={tuple(X_test.shape)}  n_features={N_FEATURES}")
    print()
    for r in range(1, 4):
        s = scenario_for_round(r)
        print(f"   preset round {r}: {s['region']} / {s['place']}  GPS {s['gps_wgs84']}")
    print()

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
