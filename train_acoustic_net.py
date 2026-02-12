import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os

# ==========================================
# 0. 辅助：确保XML文件存在
# ==========================================
XML_FILENAME = "fibonacci_32_user.xml"

def ensure_xml_exists():
    if not os.path.exists(XML_FILENAME):
        print(f"Creating default {XML_FILENAME}...")
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
</MicArray>
"""
        with open(XML_FILENAME, "w", encoding="utf-8") as f:
            f.write(xml_content.strip())

# ==========================================
# 1. 物理仿真引擎 (增强版 - 非相干源)
# ==========================================
class AcousticSimulator:
    def __init__(self, xml_path, n_mics=32, freq=8000, c=343, grid_res=0.01): 
        self.n_mics = n_mics
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.k = self.omega / c
        self.c = c
        
        # 解析 XML
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pos_list = root.findall('pos')
        
        x_coords = [float(p.get('x')) for p in pos_list]
        y_coords = [float(p.get('y')) for p in pos_list]
        z_coords = [float(p.get('z')) for p in pos_list]
            
        self.mic_x = np.array(x_coords)
        self.mic_y = np.array(y_coords)
        self.mic_z = np.array(z_coords)
        self.mics = np.stack([self.mic_x, self.mic_y, self.mic_z], axis=1)
        
        print(f"Loaded {len(pos_list)} mics.")
        
        # 扫描网格
        self.grid_vec = np.arange(-0.2, 0.2 + 0.0001, grid_res)
        self.grid_x, self.grid_y = np.meshgrid(self.grid_vec, self.grid_vec)
        self.n_grid = self.grid_x.size
        self.scan_z = -0.3
        
        self.grid_points = np.stack([self.grid_x.flatten(), self.grid_y.flatten(), np.full(self.n_grid, self.scan_z)], axis=1)

    def generate_batch(self, batch_size=32, snr=20):
        """生成训练数据 (Incoherent Sources - 与MATLAB一致)"""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # 随机声源数量 1~4
            rand_val = np.random.rand()
            if rand_val < 0.3: n_sources = 1
            elif rand_val < 0.7: n_sources = 2
            elif rand_val < 0.9: n_sources = 3
            else: n_sources = 4

            # [关键修改]：初始化 CSM 矩阵，而不是复声压向量
            # 我们直接在 CSM 层面进行叠加，模拟非相干源
            csm_accum = np.zeros((self.n_mics, self.n_mics), dtype=complex)
            target_map = np.zeros_like(self.grid_x.flatten())
            
            existing_sources = []

            for _ in range(n_sources):
                valid_pos = False
                tries = 0
                while not valid_pos and tries < 10:
                    sx = np.random.uniform(-0.15, 0.15)
                    sy = np.random.uniform(-0.15, 0.15)
                    sz = self.scan_z
                    if len(existing_sources) == 0:
                        valid_pos = True
                    else:
                        dists = [np.linalg.norm(np.array([sx, sy]) - np.array([ex, ey])) for ex, ey in existing_sources]
                        if min(dists) > 0.05: # 至少间隔 5cm
                            valid_pos = True
                    tries += 1
                
                if valid_pos:
                    existing_sources.append((sx, sy))
                    src_pos = np.array([sx, sy, sz])
                    dists = np.linalg.norm(self.mics - src_pos, axis=1)
                    
                    # 随机相位和幅度
                    amp = np.random.uniform(0.2, 1.5) 
                    phase = np.random.uniform(0, 2*np.pi)
                    
                    # 计算单个声源的信号向量
                    signal = amp * (1/dists) * np.exp(-1j * (self.k * dists + phase))
                    
                    # [关键修改]：直接累加 CSM (S * S')
                    # 这消除了不同声源之间的干涉项 (Cross-terms)，匹配 MATLAB 的仿真逻辑
                    csm_accum += np.outer(signal, signal.conj())
                    
                    # 生成 Target (高斯核)
                    dists_grid = np.linalg.norm(self.grid_points - src_pos, axis=1)
                    gaussian_blob = np.exp(- (dists_grid / 0.025)**2)
                    target_map = np.maximum(target_map, gaussian_blob)

            # CSM 后处理
            csm = csm_accum
            
            trace_val = np.trace(csm).real
            if trace_val < 1e-9: trace_val = 1e-9
            
            # 加噪
            current_snr = np.random.uniform(10, 30)
            noise_power = trace_val / self.n_mics / (10**(current_snr/10))
            csm += noise_power * np.eye(self.n_mics) * (np.random.randn() + 1j*np.random.randn())
            
            # 归一化 (与 MATLAB 一致)
            csm = csm / np.trace(csm).real
            
            csm_input = np.stack([csm.real, csm.imag], axis=0)
            target_map = np.clip(target_map, 0, 1)
            
            inputs.append(csm_input)
            targets.append(target_map.reshape(self.grid_x.shape))
            
        return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

# ==========================================
# 2. ResNet 模型定义 (保持不变)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AcousticResNet(nn.Module):
    def __init__(self, input_channels=2, output_size=(41, 41)):
        super(AcousticResNet, self).__init__()
        self.output_flat_size = output_size[0] * output_size[1]
        
        self.entry = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = ResBlock(64)
        self.layer2 = ResBlock(64)
        self.layer3 = ResBlock(64)
        self.layer4 = ResBlock(64)
        self.final_conv = nn.Conv2d(64, 8, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 32 * 32, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_flat_size),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        x = self.fc(x)
        return x.view(-1, 41, 41)

# ==========================================
# 3. 训练流程
# ==========================================
def train():
    ensure_xml_exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    simulator = AcousticSimulator(xml_path=XML_FILENAME, grid_res=0.01)
    
    model = AcousticResNet(output_size=(41, 41)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() 
    
    losses = []
    
    print(f"Starting Training (Mode: Incoherent Sources)...")
    model.train()
    
    for epoch in tqdm(range(500)):
        inputs, targets = simulator.generate_batch(batch_size=32, snr=20)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss (Incoherent)")
    plt.show()
    
    # 保存
    torch.save(model.state_dict(), "acoustic_resnet_weights_sharp.pth")
    print("Model saved to acoustic_resnet_weights_sharp.pth")

if __name__ == "__main__":
    train()