"""APC 决策规则
所有规则集中在此文件，修改规则只需改这里
"""

from typing import Dict, Any
from models import Context


class ParticipationRule:
    """规则1：决定是否参与 FL 轮次"""
    
    def __init__(self, config: Dict[str, float]):
        self.min_trust = config.get("min_trust_score", 0.3)
        self.min_resource = config.get("min_resource", 0.2)
    
    def decide(self, ctx: Context) -> int:
        """
        返回 0（不参与）或 1（参与）
        
        逻辑：信任分太低 或 资源太低 → 不参与
        """
        if ctx.trust_score < self.min_trust:
            return 0
        if ctx.resource_availability < self.min_resource:
            return 0
        return 1


class FrequencyRule:
    """规则2：决定参与频率（每 N 轮参与一次）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.low_resource_th = config.get("low_resource_threshold", 0.3)
        self.high_threat_th = config.get("high_threat_threshold", 0.7)
        self.high_criticality_th = config.get("high_criticality_threshold", 0.8)
        
        self.high_freq = config.get("high_frequency", 1)
        self.medium_freq = config.get("medium_frequency", 3)
        self.low_freq = config.get("low_frequency", 4)
        self.default_freq = config.get("default_frequency", 2)
    
    def decide(self, ctx: Context) -> int:
        """
        频率值越小表示越频繁（1=每轮，3=每3轮）
        
        优先级：资源紧张 > 威胁高 > 任务关键 > 默认
        """
        # 资源紧张 → 低频（省电）
        if ctx.resource_availability < self.low_resource_th:
            return self.low_freq
        
        # 威胁高 → 中频（减少暴露）
        if ctx.threat_level > self.high_threat_th:
            return self.medium_freq
        
        # 任务关键 → 高频（保证模型质量）
        if ctx.mission_criticality > self.high_criticality_th:
            return self.high_freq
        
        return self.default_freq


class NoiseRule:
    """规则3：决定噪声尺度（基于差分隐私 epsilon）"""
    
    def __init__(self, config: Dict[str, float]):
        self.base_epsilon = config.get("base_epsilon", 1.0)
        self.min_epsilon = config.get("min_epsilon", 0.2)
        self.max_epsilon = config.get("max_epsilon", 1.5)
        
        self.w_threat = config.get("weight_threat", 0.3)
        self.w_trust = config.get("weight_trust", 0.2)
        self.w_criticality = config.get("weight_criticality", 0.2)
        self.w_resource = config.get("weight_resource", 0.1)
    
    def decide(self, ctx: Context) -> float:
        """
        计算 epsilon，然后 noise_scale = 1 / epsilon
        
        公式：
        epsilon = base * (1 - w_threat*threat - w_trust*(1-trust) 
                         + w_criticality*criticality - w_resource*(1-resource))
        
        noise_scale 越大 = 隐私保护越强
        """
        # 威胁高 → 降低 epsilon
        threat_penalty = self.w_threat * ctx.threat_level
        
        # 信任低 → 降低 epsilon
        trust_penalty = self.w_trust * (1 - ctx.trust_score)
        
        # 任务关键 → 提高 epsilon（保持精度）
        criticality_bonus = self.w_criticality * ctx.mission_criticality
        
        # 资源低 → 降低 epsilon（减小计算开销）
        resource_penalty = self.w_resource * (1 - ctx.resource_availability)
        
        epsilon = self.base_epsilon * (1 - threat_penalty - trust_penalty 
                                        + criticality_bonus - resource_penalty)
        
        # 裁剪到安全范围
        epsilon = max(self.min_epsilon, min(self.max_epsilon, epsilon))
        
        # noise_scale = 1 / epsilon（epsilon 越小 → 噪声越大）
        noise_scale = 1.0 / epsilon
        
        # 限制噪声范围 [0.1, 2.0]
        noise_scale = max(0.1, min(2.0, noise_scale))
        
        return noise_scale