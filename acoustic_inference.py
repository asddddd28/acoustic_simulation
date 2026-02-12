# acoustic_inference.py
import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================
# 1. 定义网络结构 (必须与训练代码一致)
# ==========================================
class MicArrayNet(nn.Module):
    def __init__(self, n_mics=32, grid_size=20):
        super(MicArrayNet, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        # 空间适配
        self.adapter = nn.Upsample(size=(grid_size, grid_size), mode='bilinear', align_corners=True)
        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adapter(x)
        out = self.head(x)
        conf = torch.sigmoid(out[:, 0:1, :, :])
        offsets = torch.tanh(out[:, 1:3, :, :]) * 0.5
        return torch.cat([conf, offsets], dim=1)

# ==========================================
# 2. 推理引擎类 (供 MATLAB 调用)
# ==========================================
class InferenceEngine:
    def __init__(self, model_path="mic_array_net.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Python Engine: Using device {self.device}")
        
        # 初始化模型
        self.model = MicArrayNet(n_mics=32, grid_size=20).to(self.device)
        
        # 加载权重
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"Python Engine: Model loaded from {model_path}")
                self.ready = True
            except Exception as e:
                print(f"Python Engine Error: Failed to load weights. {e}")
                self.ready = False
        else:
            print(f"Python Engine Error: File not found {model_path}")
            self.ready = False

    def predict(self, csm_real_flat, csm_imag_flat, rows, cols):
        """
        MATLAB 传来的数据通常是展平的 (1D array) 或者 list
        rows, cols: 矩阵的维度 (32, 32)
        """
        if not self.ready:
            return []

        try:
            # 1. 重塑数据 (MATLAB 是一维传入，需 reshape)
            # 注意 MATLAB 是列优先 (F-order)，但通常 numpy处理行优先
            # 我们先转成 numpy array
            real = np.array(csm_real_flat, dtype=np.float32).reshape(rows, cols)
            imag = np.array(csm_imag_flat, dtype=np.float32).reshape(rows, cols)
            
            # 2. 构造 Tensor (Batch=1, Channel=2, H, W)
            inp = np.stack([real, imag], axis=0) # (2, 32, 32)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(self.device)
            
            # 3. 推理
            with torch.no_grad():
                out = self.model(inp_tensor)
                out = out.cpu().numpy()[0] # (3, 20, 20)
            
            # 4. 解析结果
            conf_map = out[0]
            off_x = out[1]
            off_y = out[2]
            
            results = []
            grid_size = 20
            scan_range = 0.25 # 必须与训练一致
            step = (scan_range * 2) / grid_size
            
            # 简单阈值过滤
            ys, xs = np.where(conf_map > 0.01) # 返回所有可能的点，阈值在MATLAB端控制
            
            for x, y in zip(xs, ys):
                dx = float(off_x[y, x])
                dy = float(off_y[y, x])
                conf = float(conf_map[y, x])
                
                # 计算物理坐标
                px = (x + 0.5 + dx) * step - scan_range
                py = (y + 0.5 + dy) * step - scan_range
                
                # [x, y, conf]
                results.append([px, py, conf])
                
            return results # 返回 List of Lists
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return []

# 全局单例 (可选，方便直接调用函数模式)
engine = None

def load_model(path):
    global engine
    engine = InferenceEngine(path)
    return engine.ready

def run_inference(real, imag, r, c):
    global engine
    if engine:
        return engine.predict(real, imag, r, c)
    return []