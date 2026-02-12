import asyncio
import websockets
import struct
import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.patches as pt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 0. 配置区域
# ==========================================
DEBUG_SAVE_IMAGES = False       # 是否保存热力图用于调试
DEBUG_FOLDER = "debug_frames"
WEIGHT_PATH = "acoustic_resnet_robust_best.pth"

# [核心定位参数]
CENTROID_RADIUS = 0.05         # 质心聚合半径 (5cm)
ENERGY_RELATIVE_THRESH = 0.15  # 局部能量容忍度 (Peak - 0.15)

if DEBUG_SAVE_IMAGES:
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ==========================================
# 1. 模型定义 (必须与训练代码完全一致)
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
            nn.Dropout(0.3), # 必须保留，即使 eval() 模式下不生效
            nn.Linear(2048, 41 * 41), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x.view(-1, 41, 41)

# ==========================================
# 2. 质心检测算法
# ==========================================
def detect_sources_by_centroid(grid_input, global_threshold, radius_m=0.05, energy_delta=0.15):
    working_grid = grid_input.copy()
    rows, cols = working_grid.shape
    
    # 物理坐标范围需与训练一致 (-0.2 到 0.2)
    x_axis = np.linspace(-0.2, 0.2, cols)
    y_axis = np.linspace(-0.2, 0.2, rows)
    X, Y = np.meshgrid(x_axis, y_axis)
    
    detected_sources = []
    detected_masks = [] 
    
    while True:
        max_idx = np.argmax(working_grid)
        max_r, max_c = np.unravel_index(max_idx, (rows, cols))
        max_val = working_grid[max_r, max_c]
        
        # 如果当前最高置信度低于阈值，停止搜索
        if max_val < global_threshold:
            break
            
        peak_x, peak_y = X[max_r, max_c], Y[max_r, max_c]
        
        # 区域生长：筛选出距离近且能量高的像素
        dist_map = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
        final_mask = (dist_map <= radius_m) & (working_grid >= (max_val - energy_delta))
        
        if not np.any(final_mask):
            working_grid[max_r, max_c] = 0 
            continue
            
        # 权重质心计算
        weights = working_grid[final_mask]
        centroid_x = np.sum(X[final_mask] * weights) / np.sum(weights)
        centroid_y = np.sum(Y[final_mask] * weights) / np.sum(weights)
            
        detected_sources.append((centroid_x, centroid_y, max_val))
        detected_masks.append(final_mask)
        
        # 消隐：清除已处理区域，寻找下一个声源
        working_grid[dist_map <= radius_m] = 0.0
        
    return detected_sources, detected_masks

# ==========================================
# 3. 调试绘图函数 (可选)
# ==========================================
def save_debug_plot(grid, sources, masks, req_id):
    try:
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(grid, origin='lower', extent=[-0.2, 0.2, -0.2, 0.2], cmap='jet', vmin=0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        
        for i, s in enumerate(sources):
            ax.scatter(s[0], s[1], c='red', edgecolors='white', s=50, zorder=10)
            ax.text(s[0]+0.01, s[1]+0.01, f"{s[2]:.2f}", color='white', fontsize=9, fontweight='bold')
            circle = pt.Circle((s[0], s[1]), CENTROID_RADIUS, color='white', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        ax.set_title(f"Req {req_id} | Detected: {len(sources)}")
        fig.savefig(os.path.join(DEBUG_FOLDER, f"req_{req_id:04d}.png"))
    except Exception as e: print(f"Plot error: {e}")

# ==========================================
# 4. WebSocket 服务
# ==========================================
async def handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_running_loop()

    # 模型加载
    model = AcousticResNet().to(device)
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        model.eval()
        print(f"✓ Robust model loaded: {WEIGHT_PATH}")
    else:
        print(f"✗ Error: {WEIGHT_PATH} not found!")
        return

    req_count = 0
    try:
        async for message in websocket:
            req_count += 1
            start_t = time.time()
            
            # 解析数据 (threshold + CSM_Real[1024] + CSM_Imag[1024])
            if len(message) != 2049 * 4: continue
            all_floats = np.frombuffer(message, dtype='<f4')
            
            threshold = all_floats[0]
            csm_real = all_floats[1:1025].reshape(32, 32, order='F')
            csm_imag = all_floats[1025:2049].reshape(32, 32, order='F')

            # 推理
            inp = torch.stack([torch.from_numpy(csm_real), torch.from_numpy(csm_imag)], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(inp).squeeze().cpu().numpy()
            
            # 定位
            sources, masks = detect_sources_by_centroid(output, threshold, CENTROID_RADIUS, ENERGY_RELATIVE_THRESH)
            
            # 日志打印
            dt = (time.time() - start_t) * 1000
            print(f"[{req_count:04d}] Found {len(sources)} sources in {dt:.1f}ms | Thr: {threshold:.2f}")

            # 调试绘图
            if DEBUG_SAVE_IMAGES:
                loop.run_in_executor(plot_executor, save_debug_plot, output.copy(), sources, masks, req_count)

            # 发送响应 (count, x1, y1, x2, y2...)
            response = struct.pack('<i', len(sources))
            for s in sources:
                response += struct.pack('<ff', float(s[0]), float(s[1]))
            await websocket.send(response)

    except websockets.exceptions.ConnectionClosed: pass
    finally: plot_executor.shutdown(wait=False)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 50005, max_size=None):
        await asyncio.Future() # run forever

if __name__ == "__main__":
    print("WebSocket Inference Server (Robust v2) started on port 50005...")
    asyncio.run(main())