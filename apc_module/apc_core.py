"""APC 核心控制器"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from models import Context, PrivacyParams
from rules import ParticipationRule, FrequencyRule, NoiseRule


class AdaptivePrivacyController:
    """自适应隐私控制器
    
    根据实时上下文动态调整隐私参数：
    - participation_level: 是否参与 FL
    - update_frequency: 参与频率
    - noise_scale: 差分隐私噪声尺度
    
    使用方式:
        apc = AdaptivePrivacyController("config.yaml")
        params = apc.decide(threat_level=0.8, 
                           mission_criticality=0.9,
                           trust_score=0.9,
                           resource_availability=0.5)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 APC
        
        Args:
            config_path: 配置文件路径，默认使用同目录下的 config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config = self._load_config(config_path)
        
        # 初始化各规则模块
        self.participation_rule = ParticipationRule(self.config.get("participation", {}))
        self.frequency_rule = FrequencyRule(self.config.get("frequency", {}))
        self.noise_rule = NoiseRule(self.config.get("noise", {}))
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def decide(self,
               threat_level: float,
               mission_criticality: float,
               trust_score: float,
               resource_availability: float) -> PrivacyParams:
        """
        根据上下文决定隐私参数
        
        Args:
            threat_level: 威胁等级 [0,1]
            mission_criticality: 任务关键性 [0,1]
            trust_score: 信任评分 [0,1]
            resource_availability: 资源可用性 [0,1]
        
        Returns:
            PrivacyParams: 隐私参数配置
        """
        # 封装输入
        ctx = Context(
            threat_level=threat_level,
            mission_criticality=mission_criticality,
            trust_score=trust_score,
            resource_availability=resource_availability
        )
        
        # 调用各规则
        participation = self.participation_rule.decide(ctx)
        frequency = self.frequency_rule.decide(ctx)
        noise = self.noise_rule.decide(ctx)
        
        return PrivacyParams(
            participation_level=participation,
            update_frequency=frequency,
            noise_scale=noise
        )
    
    def decide_with_context(self, ctx: Context) -> PrivacyParams:
        """使用 Context 对象作为输入"""
        return self.decide(
            threat_level=ctx.threat_level,
            mission_criticality=ctx.mission_criticality,
            trust_score=ctx.trust_score,
            resource_availability=ctx.resource_availability
        )