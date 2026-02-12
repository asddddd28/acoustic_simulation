import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os

# ==========================================
# 0. 辅助：麦克风阵列配置 (Fibonacci 32)
# ==========================================
XML_FILENAME = "fibonacci_32_user.xml"

def ensure_xml_exists():
    """生成麦克风阵列 XML 配置文件"""
    if not os.path.exists(XML_FILENAME):
        xml_content = """<?xml version="1.0" ?>
<MicArray name="Fibonacci_32_User">
  <pos Name="U1" x="-0.018430" y="0.016890" z="0.0"/>
  <pos Name="U17" x="-0.093170" y="0.003850" z="0.0"/>
  <pos Name="U2" x="0.003680" y="-0.041900" z="0.0"/>
  <pos Name="U18" x="0.067590" y="-0.067260" z="0.0"/>
  <pos Name="U3" x="0.029890" y="0.038990" z="0.0"/>
  <pos Name="U19" x="-0.004500" y="0.097290" z="0.0"/>
  <pos Name="U4" x="-0.053720" y="-0.009500" z="0.0"/>
  <pos Name="U20" x="-0.063670" y="-0.076300" z="0.0"/>
  <pos Name="U5" x="0.049890" y="-0.031730" z="0.0"/>
  <pos Name="U21" x="0.100400" y="0.013510" z="0.0"/>
  <pos Name="U6" x="-0.016390" y="0.060990" z="0.0"/>
  <pos Name="U22" x="-0.084700" y="0.058940" z="0.0"/>
  <pos Name="U7" x="-0.030790" y="-0.059280" z="0.0"/>
  <pos Name="U23" x="0.023050" y="-0.102470" z="0.0"/>
  <pos Name="U8" x="0.065890" y="0.024060" z="0.0"/>
  <pos Name="U24" x="0.053110" y="0.092690" z="0.0"/>
  <pos Name="U9" x="-0.067720" y="0.027950" z="0.0"/>
  <pos Name="U25" x="-0.103450" y="-0.033000" z="0.0"/>
  <pos Name="U10" x="0.032290" y="-0.069010" z="0.0"/>
  <pos Name="U26" x="0.100140" y="-0.046270" z="0.0"/>
  <pos Name="U11" x="0.023630" y="0.075340" z="0.0"/>
  <pos Name="U27" x="-0.043240" y="0.103320" z="0.0"/>
  <pos Name="U12" x="-0.070590" y="-0.040910" z="0.0"/>
  <pos Name="U28" x="-0.038470" y="-0.106950" z="0.0"/>
  <pos Name="U13" x="0.082140" y="-0.018060" z="0.0"/>
  <pos Name="U29" x="0.102050" y="0.053630" z="0.0"/>
  <pos Name="U14" x="-0.049760" y="0.070780" z="0.0"/>
  <pos Name="U30" x="-0.113020" y="0.029790" z="0.0"/>
  <pos Name="U15" x="-0.011420" y="-0.088110" z="0.0"/>
  <pos Name="U31" x="0.064070" y="-0.099640" z="0.0"/>
  <pos Name="U16" x="0.069650" y="0.058700" z="0.0"/>
  <pos Name="U32" x="0.020330" y="0.118270" z="0.0"/>
</MicArray>"""
        with open(XML_FILENAME, "w", encoding="utf-8") as f:
            f.write(xml_content.strip())

# ==========================================
# 1. 物理仿真引擎 (增强数据集生成)
# ==========================================
class AcousticSimulator:
    def __init__(self, xml_path, n_mics=32, freq=8000, c=343, grid_res=0.01, z_range=(-0.2, -0.45)): 
        self.n_mics = n_mics
        self.freq = freq
        self.k = (2 * np.pi * freq) / c
        self.z_range = z_range # 深度随机化范围
        
        # 加载阵列坐标
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pos_list = root.findall('pos')
        self.mics = np.array([[float(p.get('x')), float(p.get('y')), float(p.get('z'))] for p in pos_list])
        
        # 扫描网格 (用于 Label)
        self.grid_vec = np.arange(-0.2, 0.2 + 0.0001, grid_res)
        self.grid_x, self.grid_y = np.meshgrid(self.grid_vec, self.grid_vec)
        self.flat_grid_x = self.grid_x.flatten()
        self.flat_grid_y = self.grid_y.flatten()

    def generate_batch(self, batch_size=32):
        """生成强化训练数据"""
        inputs, targets = [], []
        for _ in range(batch_size):
            # 难例挖掘：随机 1~5 个声源
            n_sources = np.random.randint(1, 6)
            csm_accum = np.zeros((self.n_mics, self.n_mics), dtype=complex)
            target_map = np.zeros_like(self.flat_grid_x)
            existing_sources = []

            for _ in range(n_sources):
                valid_pos, tries = False, 0
                while not valid_pos and tries < 15:
                    sx, sy = np.random.uniform(-0.18, 0.18, 2)
                    sz = np.random.uniform(self.z_range[0], self.z_range[1]) # 随机深度
                    if not existing_sources or min([np.linalg.norm(np.array([sx, sy]) - np.array(e)) for e in existing_sources]) > 0.03:
                        valid_pos = True
                    tries += 1
                
                if valid_pos:
                    existing_sources.append((sx, sy))
                    src_pos = np.array([sx, sy, sz])
                    dists_mic = np.linalg.norm(self.mics - src_pos, axis=1)
                    
                    # 动态幅度与非相干叠加
                    amp = np.random.uniform(0.3, 2.0)
                    signal = amp * (1/dists_mic) * np.exp(-1j * (self.k * dists_mic + np.random.uniform(0, 2*np.pi)))
                    csm_accum += np.outer(signal, signal.conj())
                    
                    # 标签锐化 (Sigma=0.015) 提高置信度峰值
                    dists_2d = np.sqrt((self.flat_grid_x - sx)**2 + (self.flat_grid_y - sy)**2)
                    target_map = np.maximum(target_map, np.exp(-(dists_2d / 0.015)**2))

            # 噪声增强 (5dB~30dB)
            snr = np.random.uniform(5, 30)
            noise_pwr = np.trace(csm_accum).real / self.n_mics / (10**(snr/10))
            csm = csm_accum + noise_pwr * np.eye(self.n_mics)
            
            csm /= np.trace(csm).real # 归一化
            inputs.append(np.stack([csm.real, csm.imag], axis=0))
            targets.append(target_map.reshape(self.grid_x.shape))
            
        return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

# ==========================================
# 2. ResNet 模型定义 (同步 Dropout)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )
    def forward(self, x): return nn.functional.relu(x + self.net(x))

class AcousticResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 8, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 32 * 32, 2048), nn.ReLU(),
            nn.Dropout(0.3), # 提升泛化能力的正则化
            nn.Linear(2048, 41 * 41), nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x).pipe(self.fc).view(-1, 41, 41)
    
    # 兼容性处理
    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x.view(-1, 41, 41)

# ==========================================
# 3. 强化训练流程
# ==========================================
def train():
    ensure_xml_exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = AcousticSimulator(XML_FILENAME)
    model = AcousticResNet().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.5)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    losses = []

    print(f"Starting Robust Training on {device}...")
    for epoch in range(1000): # 增加训练轮次
        model.train()
        epoch_loss = 0
        batches = 64 # 每轮 64 个 batch
        
        for _ in range(batches):
            inputs, targets = simulator.generate_batch(32)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches
        scheduler.step(avg_loss)
        losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "acoustic_resnet_robust_best.pth") # 保存最优模型

    # 绘制损失曲线
    plt.plot(losses); plt.title("Robust Training Loss"); plt.yscale('log'); plt.show()

if __name__ == "__main__":
    train()