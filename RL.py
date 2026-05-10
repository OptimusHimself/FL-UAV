import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# =====================================
# 参数
# =====================================
num_clients = 100
state_dim = 5
episodes = 300
batch_size = 64
gamma = 0.95
lr = 1e-3
buffer_size = 5000
target_update = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# Replay Buffer
# =====================================
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, s, r, s_next):
        self.buffer.append((s, r, s_next))

    def sample(self, batch):
        data = random.sample(self.buffer, batch)
        s, r, s_next = zip(*data)
        return np.array(s), np.array(r), np.array(s_next)

    def __len__(self):
        return len(self.buffer)

memory = ReplayBuffer()

# =====================================
# 网络：输出一个评分
# =====================================
class TrustNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = TrustNet().to(device)
target_model = TrustNet().to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# =====================================
# 初始化状态
# [质量 延迟 loss 电量 攻击]
# =====================================
def init_state():
    return np.stack([
        np.random.rand(num_clients),
        np.random.rand(num_clients),
        np.random.rand(num_clients),
        np.random.rand(num_clients),
        np.random.randint(0,2,num_clients)
    ], axis=1)

# =====================================
# 环境变化
# =====================================
def update_state(state):
    s = state.copy()

    s[:,1] = np.clip(s[:,1] + np.random.normal(0,0.05,num_clients),0,1)
    s[:,2] = np.clip(s[:,2] + np.random.normal(0,0.05,num_clients),0,1)
    s[:,3] = np.clip(s[:,3] - np.random.uniform(0.01,0.03,num_clients),0,1)
    s[:,4] = np.random.randint(0,2,num_clients)

    return s

# =====================================
# Reward（真实评分）
# =====================================
def reward_function(s):
    dq, lat, loss, energy, attack = s

    reward = (
        2.0*dq
        -1.2*lat
        -0.8*loss
        +1.0*energy
        -2.0*attack
    )

    return reward

# =====================================
# 训练
# =====================================
state = init_state()

for ep in range(episodes):

    next_state = update_state(state)

    for i in range(num_clients):

        s = state[i]
        s_next = next_state[i]
        r = reward_function(s)

        memory.push(s, r, s_next)

    state = next_state

    if len(memory) >= batch_size:

        s_batch, r_batch, s_next_batch = memory.sample(batch_size)

        s_batch = torch.FloatTensor(s_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
        s_next_batch = torch.FloatTensor(s_next_batch).to(device)

        q = model(s_batch)

        with torch.no_grad():
            next_q = target_model(s_next_batch)
            y = r_batch + gamma * next_q

        loss = loss_fn(q, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    if (ep+1)%20==0:
        print(f"Episode {ep+1}")

# =====================================
# 最终评分
# =====================================
model.eval()

with torch.no_grad():

    scores = model(torch.FloatTensor(state).to(device)).cpu().numpy().flatten()

# 归一化到0~100
scores = (scores - scores.min()) / (scores.max()-scores.min()+1e-8)
scores = scores * 100

print("\n======= 所有无人机信任评分 =======")

for i in range(num_clients):
    print(f"UAV {i:03d}: {scores[i]:.2f}")