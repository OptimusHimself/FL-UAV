"""数据模型定义"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Context:
    """APC 输入：来自 RL 模块的上下文"""
    threat_level: float          # 威胁等级 [0,1]
    mission_criticality: float   # 任务关键性 [0,1]
    trust_score: float           # 信任评分 [0,1]
    resource_availability: float # 资源可用性 [0,1]
    
    def __post_init__(self):
        """自动裁剪输入范围"""
        self.threat_level = max(0.0, min(1.0, self.threat_level))
        self.mission_criticality = max(0.0, min(1.0, self.mission_criticality))
        self.trust_score = max(0.0, min(1.0, self.trust_score))
        self.resource_availability = max(0.0, min(1.0, self.resource_availability))


@dataclass
class PrivacyParams:
    """APC 输出：隐私参数"""
    participation_level: int    # 0 或 1，是否参与本轮 FL
    update_frequency: int       # 1-10，每 N 轮参与一次
    noise_scale: float          # 0.1-2.0，差分隐私噪声尺度
    
    def __repr__(self) -> str:
        return (f"PrivacyParams(participation={self.participation_level}, "
                f"freq={self.update_frequency}, noise={self.noise_scale:.3f})")