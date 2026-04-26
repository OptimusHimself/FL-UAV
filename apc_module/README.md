# APC Module - Adaptive Privacy Controller

自适应隐私控制器模块，用于无人机集群联邦学习系统中动态调整隐私参数。
// 环境使用 nn-zero-to-hero

## 功能概述

APC 模块根据实时上下文（威胁等级、任务关键性、信任评分、资源可用性）动态决定：
- **participation_level**: 是否参与当前 FL 轮次
- **update_frequency**: 参与频率（每 N 轮参与一次）
- **noise_scale**: 差分隐私噪声尺度

## 输入输出

### 输入（来自 RL 模块）

| 参数 | 类型 | 范围 | 描述 |
|------|------|------|------|
| `threat_level` | float | 0-1 | 威胁等级，越高越危险 |
| `mission_criticality` | float | 0-1 | 任务关键性，越高越重要 |
| `trust_score` | float | 0-1 | 信任评分，越低越可疑 |
| `resource_availability` | float | 0-1 | 资源可用性，越低越紧张 |

### 输出（给 FL / IDS 模块）

| 参数 | 类型 | 范围 | 描述 |
|------|------|------|------|
| `participation_level` | int | 0 或 1 | 0=不参与，1=参与 |
| `update_frequency` | int | 1-10 | 每 N 轮参与一次 |
| `noise_scale` | float | 0.1-2.0 | 噪声尺度，越大隐私越强 |

## 决策规则

### 规则1：参与度 (participation_level)
if trust_score < 0.3:
participation_level = 0
elif resource_availability < 0.2:
participation_level = 0
else:
participation_level = 1


### 规则2：更新频率 (update_frequency)
if resource_availability < 0.3:
update_frequency = 4 # 低频，省电
elif threat_level > 0.7:
update_frequency = 3 # 中频，减少暴露
elif mission_criticality > 0.8:
update_frequency = 1 # 高频，保证精度
else:
update_frequency = 2 # 默认

### 规则3：噪声尺度 (noise_scale)

基于差分隐私的 epsilon 计算：
epsilon = base * (1 - w_threatthreat - w_trust(1-trust)

w_criticalitycriticality - w_resource(1-resource))

noise_scale = 1 / epsilon



- 威胁高、信任低、资源低 → epsilon 降低 → 噪声增大
- 任务关键 → epsilon 提高 → 噪声减小

**权重配置**（在 config.yaml 中调整）：
- threat: 0.3
- trust: 0.2
- criticality: 0.2
- resource: 0.1

## 环境配置

### 依赖安装

```bash
pip install pyyaml
```

文件结构

```text
apc_module/
├── __init__.py          # 模块入口
├── apc_core.py          # 核心控制器
├── rules.py             # 规则定义（修改规则改这里）
├── config.yaml          # 配置文件（修改阈值改这里）
├── models.py            # 数据模型
├── test_apc.py          # 单元测试
├── example.py           # 使用示例
└── README.md            # 本文档
```

# 快速开始
## 基础使用

```python

from apc_module import AdaptivePrivacyController

# 初始化
apc = AdaptivePrivacyController()

# 决策
params = apc.decide(
    threat_level=0.8,
    mission_criticality=0.9,
    trust_score=0.9,
    resource_availability=0.5
)

print(params)
# 输出: PrivacyParams(participation=1, freq=1, noise=0.667)
```
## 使用 Context 对象

```python
from apc_module import AdaptivePrivacyController, Context

apc = AdaptivePrivacyController()
ctx = Context(threat_level=0.6, mission_criticality=0.7,
              trust_score=0.5, resource_availability=0.4)
params = apc.decide_with_context(ctx)
```


## 修改规则  
### 方式1：修改配置文件（推荐）
编辑 config.yaml，修改阈值即可：

```yaml
participation:
  min_trust_score: 0.3    # 改这里
  min_resource: 0.2       # 改这里
```

### 方式2：修改规则逻辑  
编辑 rules.py，找到对应规则类修改决策逻辑。

## 运行测试

```bash
python test_apc.py
```

## 运行示例
```bash
python example.py
```

## 与其他模块集成  
### 在 FL 模块中使用
```python
from apc_module import AdaptivePrivacyController

class FederatedLearning:
    def __init__(self):
        self.apc = AdaptivePrivacyController()
        self.uavs = {}  # UAV 状态
    
    def select_participants(self):
        participants = []
        for uav_id, state in self.uavs.items():
            params = self.apc.decide(
                state.threat_level,
                state.mission_criticality,
                state.trust_score,
                state.resource_availability
            )
            if params.participation_level == 1:
                participants.append(uav_id)
        return participants
```

### 在 IDS 模块中使用
```python
from apc_module import AdaptivePrivacyController

class IntrusionDetection:
    def __init__(self):
        self.apc = AdaptivePrivacyController()
    
    def train_local_model(self, context):
        params = self.apc.decide_with_context(context)
        # 使用 params.noise_scale 进行 DP 训练
        self.train_with_dp(noise_scale=params.noise_scale)
```
# 扩展指南
## 添加新规则
在 rules.py 中新建规则类

在 apc_core.py 的 __init__ 中初始化

在 decide 方法中调用

## 添加新输入参数
修改 models.py 中的 Context 类

修改 rules.py 中的规则类

更新 config.yaml 配置


---

## 总结

| 需求 | 实现方式 |
|------|----------|
| 快速二次开发 | 规则与逻辑分离，修改 `rules.py` 或 `config.yaml` |
| 多文件结构 | 7 个文件 + 1 个 README，职责单一 |
| 路径设置 | 使用 `Path(__file__).parent` 自动定位配置文件 |
| 规则文档 | README 详细说明三条规则 |
| 环境配置 | 只需 `pip install pyyaml` |

将以上文件按结构放置后，运行 `python example.py` 即可看到输出。
