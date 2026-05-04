import torch
from collections import OrderedDict
from typing import List, Dict


def fed_avg(client_updates: List[Dict],
            strategy: str = "weighted") -> OrderedDict:
    """Federated Averaging aggregation.

    Each client_update: {
        "state_dict": OrderedDict,
        "num_samples": int
    }
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    if strategy == "weighted":
        total_samples = sum(u["num_samples"] for u in client_updates)
        weights = [u["num_samples"] / total_samples for u in client_updates]
    else:
        weights = [1.0 / len(client_updates)] * len(client_updates)

    aggregated = OrderedDict()
    keys = client_updates[0]["state_dict"].keys()

    for key in keys:
        aggregated[key] = torch.zeros_like(
            client_updates[0]["state_dict"][key], dtype=torch.float32
        )
        for i, update in enumerate(client_updates):
            aggregated[key] += weights[i] * update["state_dict"][key].float()

    return aggregated


def fed_prox_penalty(local_state: OrderedDict,
                     global_state: OrderedDict,
                     mu: float = 0.01) -> torch.Tensor:
    """Compute FedProx proximal penalty term."""
    penalty = torch.tensor(0.0)
    for key in global_state:
        penalty += ((local_state[key].float() - global_state[key].float()) ** 2).sum()
    return 0.5 * mu * penalty
