import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict

from privacy.differential_privacy import DPSGD, add_noise_to_predictions


class FLClient:
    """UAV客户端：本地CNN训练与入侵检测。

    数据流 UAV → SERVER:
      - Attack type (with DP noise): 带差分隐私噪声的攻击类型检测结果
      - Local parameter: 本地CNN训练后的模型参数（梯度/权重）
      - UAV situation parameters: 电池电量、GPS信息、CPU使用率等设备状态
    """

    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, device: torch.device,
                 learning_rate: float = 0.001,
                 clip_bound: float = 1.0,
                 noise_multiplier: float = 1.0):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate
        self.clip_bound = clip_bound
        self.noise_multiplier = noise_multiplier

    def receive_global_model(self, state_dict: OrderedDict):
        """接收SERVER下发的全局模型参数。"""
        self.model.load_state_dict(state_dict)

    def local_train(self, num_epochs: int = 3,
                    noise_scale: float = 1.0) -> Dict:
        """本地DP-SGD训练，返回上传给SERVER的数据。

        Returns:
            {
                "client_id": int,
                "local_parameter": OrderedDict,  本地模型参数
                "attack_type": ndarray,           攻击类型预测(含DP噪声)
                "uav_situation": dict,            UAV状态参数
                "num_samples": int,
                "train_loss": float,
                "train_accuracy": float,
            }
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)
        dp = DPSGD(self.model, optimizer, self.clip_bound, noise_scale)
        criterion = nn.CrossEntropyLoss()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds = []

        for epoch in range(num_epochs):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                dp.step(loss)

                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        local_param = OrderedDict(
            {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        )

        attack_preds = np.array(all_preds[:min(200, len(all_preds))])
        attack_type_dp = add_noise_to_predictions(
            attack_preds, noise_scale, num_classes=5
        )

        return {
            "client_id": self.client_id,
            "local_parameter": local_param,
            "attack_type": attack_type_dp,
            "uav_situation": {},
            "num_samples": total_samples,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "gradient_norm": 1.0,
            "data_quality": 0.7,
            "communication_reliability": 0.9,
        }

    def predict(self, data_loader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        return {
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
        }

    def predict_attack_types_with_dp(self, data_loader,
                                      noise_scale: float = 1.0,
                                      num_classes: int = 5) -> np.ndarray:
        preds = self.predict(data_loader)["predictions"]
        return add_noise_to_predictions(preds, noise_scale, num_classes)
