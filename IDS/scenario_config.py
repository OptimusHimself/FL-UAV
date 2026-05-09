"""
Federated learning round scenario: deployment theatre, anchor GPS, narrative.

Used by server (logging / eval banner) and clients (local training context).
Round 1–3 map to Iran, Sudan, Beijing; additional rounds cycle the same three.
"""

from __future__ import annotations

from typing import Any

# Telemetry channels appended after 44-d IDS features (n_features == 47).
TELEM_BATTERY_IDX = 44
TELEM_GPS_IDX = 45
TELEM_CPU_IDX = 46

SCENARIOS: dict[int, dict[str, Any]] = {
    1: {
        "region": "Middle East",
        "place": "Iran",
        "latitude_deg": 35.6892,
        "longitude_deg": 51.3890,
        "gps_wgs84": "35.6892°N, 51.3890°E",
        "training_scene": (
            "Tehran-area urban–mountain fringe patrol: IDS windows under intermittent GNSS "
            "multipath and elevated RF interference; federated round emphasizes jamming / "
            "de-authentication patterns on narrowband UAV relay links."
        ),
    },
    2: {
        "region": "North Africa",
        "place": "Sudan",
        "latitude_deg": 15.5007,
        "longitude_deg": 32.5599,
        "gps_wgs84": "15.5007°N, 32.5599°E",
        "training_scene": (
            "Khartoum-area relay UAVs over arid corridors: mixed reconnaissance and scanning "
            "traffic with sporadic UDP-flooding bursts; FL round stresses long-range lossy "
            "backhaul and heterogeneous client shards."
        ),
    },
    3: {
        "region": "East Asia",
        "place": "Beijing, China",
        "latitude_deg": 39.9042,
        "longitude_deg": 116.4074,
        "gps_wgs84": "39.9042°N, 116.4074°E",
        "training_scene": (
            "Urban core electromagnetic clutter: dense Wi-Fi/BT and AP diversity; federated "
            "round targets DDoS / brute-force-like bursts embedded in high-volume legitimate "
            "traffic on edge IDS sliding windows."
        ),
    },
}


def scenario_for_round(server_round: int) -> dict[str, Any]:
    r = max(1, int(server_round))
    key = ((r - 1) % len(SCENARIOS)) + 1
    return SCENARIOS[key]


def format_scenario_block(sc: dict[str, Any], *, title: str | None = None) -> str:
    head = title or "Deployment scenario"
    lines = [
        head,
        f"  region      : {sc['region']} — {sc['place']}",
        f"  GPS (WGS84): {sc['gps_wgs84']}  (lat={sc['latitude_deg']:.4f}°, lon={sc['longitude_deg']:.4f}°)",
        f"  scene       : {sc['training_scene']}",
    ]
    return "\n".join(lines)
