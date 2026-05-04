import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

from models.cnn_ids import CNNIDS
from fl.aggregation import fed_avg
from utils.metrics import MetricsTracker


class FLServer:
    """FL服务器：联邦学习聚合中心。

    数据流:
      - 接收UAV上传: attack_type(DP噪声), local_parameter, UAV_situation
      - 接收RL反馈: model_performance_consistency, historical_participation_rate,
                     contextual_risk, resource_fingerprinting
      - 下发给UAV: global_model, noise_scale, update_frequency
      - 发送给APC: update_frequency(arr), participation_level(arr), noise_scale(arr)
    """

    def __init__(self, model: CNNIDS, num_classes: int = 5,
                 device: torch.device = None):
        self.global_model = model
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.metrics = MetricsTracker(num_classes)
        self.participation_history: Dict[int, List[bool]] = {}

        self.current_noise_scale = np.ones(5) * 1.0
        self.current_update_frequency = np.ones(5, dtype=int) * 3
        self.current_participation_level = np.ones(5) * 0.5

    def get_global_model(self) -> OrderedDict:
        return OrderedDict(
            {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        )

    def select_clients(self, participation_levels: Dict[int, float],
                       seed: int = None) -> List[int]:
        rng = np.random.RandomState(seed)
        selected = []
        for cid, level in participation_levels.items():
            if rng.random() < level:
                selected.append(cid)
        if not selected:
            selected = [max(participation_levels, key=participation_levels.get)]
        return selected

    def aggregate(self, client_updates: List[Dict],
                  strategy: str = "weighted",
                  global_noise_scale: float = 0.0) -> OrderedDict:
        aggregated = fed_avg(client_updates, strategy)

        if global_noise_scale > 0:
            for key in aggregated:
                noise = torch.normal(
                    0, global_noise_scale,
                    size=aggregated[key].shape
                )
                aggregated[key] = aggregated[key] + noise

        self.global_model.load_state_dict(aggregated)
        return aggregated

    def evaluate(self, test_loader) -> Dict:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.global_model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = self.metrics.compute_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics

    def record_participation(self, round_num: int,
                              selected_clients: List[int],
                              total_clients: int):
        for cid in range(total_clients):
            if cid not in self.participation_history:
                self.participation_history[cid] = []
            self.participation_history[cid].append(cid in selected_clients)

    def update_rl_feedback(self, rl_feedback: Dict,
                            num_clients: int):
        """接收RL模块的反馈，更新服务器内部策略。

        RL反馈:
          - model_performance_consistency: 模型性能一致性
          - historical_participation_rate(arr): 历史参与率
          - contextual_risk: 上下文风险
          - resource_fingerprinting(arr): 资源指纹
        """
        participation_rate = rl_feedback.get(
            "historical_participation_rate",
            np.ones(num_clients) * 0.5
        )
        self.current_participation_level = np.clip(participation_rate, 0.2, 1.0)

        risk = rl_feedback.get("contextual_risk", 0.3)
        base_noise = 1.0 + risk * 2.0
        resource_fp = rl_feedback.get(
            "resource_fingerprinting",
            np.ones(num_clients) * 0.5
        )
        self.current_noise_scale = np.clip(
            base_noise * (1.0 + 0.5 * (1.0 - resource_fp)),
            0.1, 5.0
        )

        consistency = rl_feedback.get("model_performance_consistency", 0.5)
        base_freq = 1 + consistency * 4
        self.current_update_frequency = np.clip(
            np.round(base_freq * resource_fp).astype(int), 1, 5
        )

    def extract_uav_state_arrays(self, updates: List[Dict],
                                  num_clients: int) -> Dict:
        """从UAV上传数据中提取RL所需的4类状态数组。

        Args:
            updates: 各UAV客户端上传的数据列表，每项包含
                     uav_situation(battery/CPU/GPS)、attack_type(DP噪声预测)
            num_clients: 客户端总数

        Returns:
            {
                "resource_availability": (N,) 从battery/CPU/GPS计算,
                "mission_criticality": (N,) 从参与度+噪声推断,
                "threat_level": (N,) 从attack_type攻击比率计算,
                "trust_scores": (N,) 从参与历史计算,
            }
        """
        resource_avail = np.zeros(num_clients)
        mission_crit = np.zeros(num_clients)
        threat_level = np.zeros(num_clients)
        trust = np.full(num_clients, 0.5)

        for update in updates:
            cid = update.get("client_id", 0)
            if cid >= num_clients:
                continue
            situation = update.get("uav_situation", {})

            # Resource Availability: 从battery/CPU/GPS计算
            battery = situation.get("battery", 50)
            cpu = situation.get("cpu_utilization", 50)
            gps = situation.get("gps_accuracy", 5)
            resource_avail[cid] = (
                0.4 * (battery / 100.0)
                + 0.35 * (1.0 - cpu / 100.0)
                + 0.25 * (1.0 - min(gps / 10.0, 1.0))
            )

            # Mission Criticality: 从参与度+噪声推断
            participation = self.current_participation_level[cid]
            noise = self.current_noise_scale[cid]
            mission_crit[cid] = 0.7 * participation + 0.3 * (1 - noise / 5)

            # Threat Level: 从attack_type预测计算攻击比率
            attack_preds = update.get("attack_type", np.array([]))
            if len(attack_preds) > 0:
                attack_ratio = float(np.mean(attack_preds != 0))
                threat_level[cid] = 0.6 * attack_ratio + 0.4 * noise / 5
            else:
                threat_level[cid] = noise / 5

            # Trust Score: 从参与历史计算
            if cid in self.participation_history and self.participation_history[cid]:
                recent = self.participation_history[cid][-10:]
                trust[cid] = float(np.mean(recent))

        return {
            "resource_availability": np.clip(resource_avail, 0.0, 1.0),
            "mission_criticality": np.clip(mission_crit, 0.0, 1.0),
            "threat_level": np.clip(threat_level, 0.0, 1.0),
            "trust_scores": np.clip(trust, 0.0, 1.0),
        }

    def get_stats_for_apc(self, num_clients: int,
                           updates: List[Dict] = None) -> Dict:
        """生成发送给APC的统计数据。

        Args:
            num_clients: 客户端数量
            updates: 本轮UAV上传数据（可选，传入时提取真实UAV状态）

        Returns:
            包含策略数组、UAV真实状态数组、模型性能指标的字典
        """
        stats = {
            "update_frequency": self.current_update_frequency[:num_clients],
            "participation_level": self.current_participation_level[:num_clients],
            "noise_scale": self.current_noise_scale[:num_clients],
        }

        # 从UAV上传数据中提取真实状态
        if updates:
            uav_state = self.extract_uav_state_arrays(updates, num_clients)
            stats.update(uav_state)

        # 附加模型性能指标
        stats["model_performance_consistency"] = self.get_model_performance_consistency()
        if self.metrics.history:
            stats["model_accuracy"] = self.metrics.history[-1]["accuracy"]
            stats["model_loss"] = self.metrics.history[-1]["loss"]

        return stats

    def get_model_performance_metrics(self) -> Dict:
        """获取当前模型性能指标，供传递给RL作为reward信号。"""
        result = {
            "model_performance_consistency": self.get_model_performance_consistency(),
        }
        if self.metrics.history:
            result["model_accuracy"] = self.metrics.history[-1]["accuracy"]
            result["model_loss"] = self.metrics.history[-1]["loss"]
        return result

    def get_model_performance_consistency(self) -> float:
        if len(self.metrics.history) < 2:
            return 0.5
        recent = [h["accuracy"] for h in self.metrics.history[-5:]]
        return float(np.clip(1.0 - np.std(recent), 0.0, 1.0))
