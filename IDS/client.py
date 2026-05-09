import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from model import CnnLstmIDS, get_model_parameters, set_model_parameters
from scenario_config import (
    TELEM_BATTERY_IDX,
    TELEM_CPU_IDX,
    format_scenario_block,
    scenario_for_round,
)
from swarm_fl_data import (
    NUM_CLIENTS,
    SWARM_DIR,
    load_meta,
    load_train_split,
    local_train_test_split,
    split_index_for_round_client,
)


class DroneClient(fl.client.NumPyClient):
    """Each FL round uses train_split_{(round-1)*3 + client_id} from data/swarm_processed/."""

    def __init__(self, client_id: int):
        self.client_id = client_id
        meta = load_meta()
        self.num_classes = int(meta["num_classes"])
        self.n_features = int(meta["n_features"])
        self.model = CnnLstmIDS(n_features=self.n_features, num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.round_count = 0
        self._train_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._num_train = 0
        self._cached_split = -1

    def _round_from_config(self, config: dict) -> int:
        v = config.get("server_round", 1)
        return int(float(v))

    def _rebuild_loaders(self, split_idx: int) -> None:
        if split_idx == self._cached_split and self._train_loader is not None:
            return
        X, y = load_train_split(split_idx)
        if len(X) < 4:
            raise ValueError(f"train_split_{split_idx} too small: {len(X)}")
        x_tr, y_tr, x_te, y_te = local_train_test_split(X, y, fraction=0.8)
        self.x_train = torch.tensor(x_tr, dtype=torch.float32)
        self.y_train = torch.tensor(y_tr, dtype=torch.long)
        self.x_test = torch.tensor(x_te, dtype=torch.float32)
        self.y_test = torch.tensor(y_te, dtype=torch.long)
        self._num_train = len(self.x_train)
        self._train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train), batch_size=32, shuffle=True
        )
        self._test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test), batch_size=32, shuffle=False
        )
        self._cached_split = split_idx

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        rnd = self._round_from_config(config)
        split_idx = split_index_for_round_client(rnd, self.client_id)
        self.round_count += 1
        sc = scenario_for_round(rnd)
        self._rebuild_loaders(split_idx)
        set_model_parameters(self.model, parameters)

        print("\n" + "=" * 56)
        print(
            format_scenario_block(
                sc,
                title=f"[CLIENT {self.client_id}] FL round={rnd}  split={split_idx}  ({SWARM_DIR})",
            )
        )

        fit_metrics: dict[str, float] = {"client_id": float(self.client_id)}
        if self.x_train.size(-1) > TELEM_CPU_IDX:
            fit_metrics["battery_soc_mean"] = float(
                self.x_train[:, :, TELEM_BATTERY_IDX].mean().item()
            )
            fit_metrics["cpu_util_mean"] = float(
                self.x_train[:, :, TELEM_CPU_IDX].mean().item()
            )
            print(
                f"  [UAV-{self.client_id}] local train telemetry (mean over windows×timesteps): "
                f"battery={fit_metrics['battery_soc_mean']:.2f}%  "
                f"CPU={fit_metrics['cpu_util_mean']:.2f}%"
            )

        self.model.train()
        assert self._train_loader is not None
        for batch_x, batch_y in self._train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

        print(f"  [CLIENT {self.client_id}] train_windows={self._num_train}  training done.")
        print("=" * 56 + "\n")

        return get_model_parameters(self.model), self._num_train, fit_metrics

    def evaluate(self, parameters, config):
        rnd = self._round_from_config(config)
        split_idx = split_index_for_round_client(rnd, self.client_id)
        sc = scenario_for_round(rnd)
        self._rebuild_loaders(split_idx)
        set_model_parameters(self.model, parameters)

        self.model.eval()
        correct = 0
        total = 0
        assert self._test_loader is not None
        with torch.no_grad():
            for batch_x, batch_y in self._test_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total else 0.0
        ev_metrics: dict[str, float] = {"accuracy": accuracy, "client_id": float(self.client_id)}
        if self.x_test.size(-1) > TELEM_CPU_IDX:
            ev_metrics["battery_soc_mean"] = float(
                self.x_test[:, :, TELEM_BATTERY_IDX].mean().item()
            )
            ev_metrics["cpu_util_mean"] = float(
                self.x_test[:, :, TELEM_CPU_IDX].mean().item()
            )
            print(
                f"[CLIENT {self.client_id}] eval  round={rnd}  split={split_idx}  "
                f"acc={accuracy:.4%}  "
                f"battery={ev_metrics['battery_soc_mean']:.2f}%  "
                f"CPU={ev_metrics['cpu_util_mean']:.2f}%  "
                f"({sc['place']})"
            )
        else:
            print(
                f"[CLIENT {self.client_id}] eval  round={rnd}  split={split_idx}  "
                f"acc={accuracy:.4%}  ({sc['place']})"
            )

        return 0.0, total, ev_metrics


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if not (0 <= client_id < NUM_CLIENTS):
        print(f"client_id must be in [0, {NUM_CLIENTS})")
        sys.exit(1)

    load_meta()
    print(f"--- client {client_id} --- swarm data: {SWARM_DIR}")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=DroneClient(client_id).to_client(),
    )
