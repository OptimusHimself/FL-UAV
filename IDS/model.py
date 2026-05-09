import torch
import torch.nn as nn


class CnnLstmIDS(nn.Module):
    """
    Windowed IDS: input (batch, T, F) — e.g. T=16 packets, F=44 features.
    Conv1d over time (channels = features), then LSTM, then classifier.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        cnn_channels: tuple[int, int] = (64, 32),
        lstm_hidden: int = 64,
    ):
        super().__init__()
        c1, c2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.float()
        z = x.permute(0, 2, 1)  # (B, F, T)
        z = self.cnn(z)  # (B, C2, T)
        z = z.permute(0, 2, 1)  # (B, T, C2)
        out, _ = self.lstm(z)
        last = out[:, -1, :]
        return self.head(last)


# 对应毕设要求：适应无人机有限的板载计算能力（KDD 扁平特征基线）
class SimpleIDS(nn.Module):
    def __init__(self, input_dim):
        super(SimpleIDS, self).__init__()
        # 定义网络结构：38 -> 32 -> 16 -> 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32), # 输入层
            nn.ReLU(),                # 激活函数
            nn.Linear(32, 16),        # 隐藏层
            nn.ReLU(),
            nn.Linear(16, 2)          # 输出层：二分类
        )

    def forward(self, x):
        # 确保输入是 Float 类型，防止数据类型报错
        return self.fc(x.float())

def get_model_parameters(model):
    """提取模型中所有需要学习的参数（Weights & Bias）"""
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_model_parameters(model, parameters):
    """【修复】将接收到的参数重新装载进模型，修正了 load_state_dict 的拼写"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    # 关键修正点：这里必须是 load_state_dict
    model.load_state_dict(state_dict, strict=True)