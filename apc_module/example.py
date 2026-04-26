"""APC 模块使用示例"""
# example.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apc_core import AdaptivePrivacyController
from models import Context

def main():
    # 初始化 APC（自动加载 config.yaml）
    apc = AdaptivePrivacyController()
    
    # 示例1：正常场景
    print("=" * 50)
    print("场景1：正常场景")
    params = apc.decide(
        threat_level=0.3,
        mission_criticality=0.5,
        trust_score=0.8,
        resource_availability=0.7
    )
    print(f"输入: threat=0.3, critical=0.5, trust=0.8, resource=0.7")
    print(f"输出: {params}\n")
    
    # 示例2：高威胁场景
    print("=" * 50)
    print("场景2：高威胁场景")
    params = apc.decide(
        threat_level=0.9,
        mission_criticality=0.5,
        trust_score=0.8,
        resource_availability=0.7
    )
    print(f"输入: threat=0.9, critical=0.5, trust=0.8, resource=0.7")
    print(f"输出: {params}\n")
    
    # 示例3：低信任场景
    print("=" * 50)
    print("场景3：低信任场景")
    params = apc.decide(
        threat_level=0.3,
        mission_criticality=0.5,
        trust_score=0.2,
        resource_availability=0.7
    )
    print(f"输入: threat=0.3, critical=0.5, trust=0.2, resource=0.7")
    print(f"输出: {params}\n")
    
    # 示例4：低资源场景
    print("=" * 50)
    print("场景4：低资源场景")
    params = apc.decide(
        threat_level=0.3,
        mission_criticality=0.5,
        trust_score=0.8,
        resource_availability=0.1
    )
    print(f"输入: threat=0.3, critical=0.5, trust=0.8, resource=0.1")
    print(f"输出: {params}\n")
    
    # 示例5：高关键性场景
    print("=" * 50)
    print("场景5：高关键性场景")
    params = apc.decide(
        threat_level=0.3,
        mission_criticality=0.95,
        trust_score=0.8,
        resource_availability=0.7
    )
    print(f"输入: threat=0.3, critical=0.95, trust=0.8, resource=0.7")
    print(f"输出: {params}\n")
    
    # 示例6：使用 Context 对象
    print("=" * 50)
    print("场景6：使用 Context 对象")
    ctx = Context(threat_level=0.6, mission_criticality=0.7, 
                  trust_score=0.5, resource_availability=0.4)
    params = apc.decide_with_context(ctx)
    print(f"输入: {ctx}")
    print(f"输出: {params}")


if __name__ == "__main__":
    main()