# FL-UAV

Federated learning (Flower) + **CNN+LSTM** intrusion detection on windowed UAV network features (`16 × 47` per sample: 44 IDS features + synthetic battery / GPS-speed / CPU telemetry).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install torch flwr pandas scikit-learn numpy
```

1. Place raw attack CSVs under `dataset/` and optional `data/UAV-Case1-Label.csv` (first line = 44 feature names). If Case1 file is missing, `preprocess_swarm_federated.py` uses a built-in column list.
2. Build `data/swarm_processed/`:

   ```bash
   python preprocess_swarm_federated.py
   python enrich_swarm_telemetry.py
   ```

3. Run FL (one terminal server, three clients):

   ```bash
   python server.py
   python client.py 0
   python client.py 1
   python client.py 2
   ```

## Layout

| Path | Role |
|------|------|
| `preprocess_swarm_federated.py` | Build train splits + global test from `dataset/` |
| `enrich_swarm_telemetry.py` | Append 3 aligned telemetry channels |
| `client.py` / `server.py` | Flower `FedAvg`, 3 rounds, rotating `train_split_*` |
| `model.py` | `CnnLstmIDS` |
| `data/swarm_processed/` | `train_split_0..8.npz`, `test.npz`, `meta.json` |

## Remote

Upstream: `https://github.com/OptimusHimself/FL-UAV`

Push (after `git remote add origin …`):

```bash
git push -u origin main
```

Use a [Personal Access Token](https://github.com/settings/tokens) or SSH key when GitHub prompts for credentials.
