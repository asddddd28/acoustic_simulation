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
# 注意：这些默认值现在由客户端通过 WebSocket 覆盖，但保留作为回退或初始值
DEFAULT_DEBUG_SAVE = False
DEFAULT_RADIUS = 0.05
DEFAULT_ENERGY = 0.15

DEBUG_FOLDER = "debug_frames"
WEIGHT_PATH = "acoustic_resnet_robust_best.pth"

if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ==========================================
# 1. 模型定义 (保持不变)
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
            nn.Dropout(0.3),
            nn.Linear(2048, 41 * 41), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x.view(-1, 41, 41)

# ==========================================
# 2. 质心检测算法 (参数化)
# ==========================================
def detect_sources_by_centroid(grid_input, global_threshold, radius_m, energy_delta):
    """
    修改版：接受 radius_m 和 energy_delta 作为参数
    """
    working_grid = grid_input.copy()
    rows, cols = working_grid.shape
    
    # 物理坐标范围 (-0.2 到 0.2)
    x_axis = np.linspace(-0.2, 0.2, cols)
    y_axis = np.linspace(-0.2, 0.2, rows)
    X, Y = np.meshgrid(x_axis, y_axis)
    
    detected_sources = []
    detected_masks = [] 
    
    while True:
        max_idx = np.argmax(working_grid)
        max_r, max_c = np.unravel_index(max_idx, (rows, cols))
        max_val = working_grid[max_r, max_c]
        
        if max_val < global_threshold:
            break
            
        peak_x, peak_y = X[max_r, max_c], Y[max_r, max_c]
        
        # 区域生长
        dist_map = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
        # 动态使用传入的 energy_delta
        final_mask = (dist_map <= radius_m) & (working_grid >= (max_val - energy_delta))
        
        if not np.any(final_mask):
            working_grid[max_r, max_c] = 0 
            continue
            
        weights = working_grid[final_mask]
        centroid_x = np.sum(X[final_mask] * weights) / np.sum(weights)
        centroid_y = np.sum(Y[final_mask] * weights) / np.sum(weights)
            
        detected_sources.append((centroid_x, centroid_y, max_val))
        detected_masks.append(final_mask)
        
        # 消隐
        working_grid[dist_map <= radius_m] = 0.0
        
    return detected_sources, detected_masks

# ==========================================
# 3. 调试绘图函数
# ==========================================
def save_debug_plot(grid, sources, masks, req_id, radius_val):
    try:
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(grid, origin='lower', extent=[-0.2, 0.2, -0.2, 0.2], cmap='jet', vmin=0, vmax=1.0)
        fig.colorbar(im, ax=ax)
        
        for i, s in enumerate(sources):
            ax.scatter(s[0], s[1], c='red', edgecolors='white', s=50, zorder=10)
            ax.text(s[0]+0.01, s[1]+0.01, f"{s[2]:.2f}", color='white', fontsize=9, fontweight='bold')
            # 绘制实际使用的半径
            circle = pt.Circle((s[0], s[1]), radius_val, color='white', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        ax.set_title(f"Req {req_id} | Detected: {len(sources)}")
        fig.savefig(os.path.join(DEBUG_FOLDER, f"req_{req_id:04d}.png"))
    except Exception as e: print(f"Plot error: {e}")

# ==========================================
# 4. WebSocket 服务 (协议更新)
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
            
            # [协议更新] 
            # 旧长度: 1 (thr) + 1024 + 1024 = 2049
            # 新长度: 1 (thr) + 1 (rad) + 1 (egy) + 1 (dbg) + 1024 + 1024 = 2052
            
            expected_floats = 1 + 1 + 1 + 1 + 1024 + 1024
            if len(message) != expected_floats * 4: 
                print(f"Warning: Payload size mismatch. Got {len(message)} bytes, expected {expected_floats * 4}")
                continue

            all_floats = np.frombuffer(message, dtype='<f4')
            
            # 1. 解析头部参数
            threshold = float(all_floats[0])
            radius_m = float(all_floats[1])
            energy_delta = float(all_floats[2])
            debug_flag = float(all_floats[3]) > 0.5  # 1.0 -> True
            
            # 2. 解析 CSM 数据 (Offset = 4)
            csm_real = all_floats[4:1028].reshape(32, 32, order='F')
            csm_imag = all_floats[1028:2052].reshape(32, 32, order='F')

            # 推理
            inp = torch.stack([torch.from_numpy(csm_real), torch.from_numpy(csm_imag)], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(inp).squeeze().cpu().numpy()
            
            # 定位 (传入动态参数)
            sources, masks = detect_sources_by_centroid(output, threshold, radius_m, energy_delta)
            
            dt = (time.time() - start_t) * 1000
            print(f"[{req_count:04d}] N={len(sources)} | T={dt:.1f}ms | Thr:{threshold:.1f} R:{radius_m:.2f} E:{energy_delta:.2f} Dbg:{int(debug_flag)}")

            # 调试绘图 (仅当 DebugFlag 为 True 时)
            if debug_flag:
                loop.run_in_executor(plot_executor, save_debug_plot, output.copy(), sources, masks, req_count, radius_m)

            # 发送响应
            response = struct.pack('<i', len(sources))
            for s in sources:
                response += struct.pack('<ff', float(s[0]), float(s[1]))
            await websocket.send(response)

    except websockets.exceptions.ConnectionClosed: pass
    finally: plot_executor.shutdown(wait=False)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 50005, max_size=None) as server:
        await asyncio.Future()

if __name__ == "__main__":
    print("WebSocket Inference Server v3 (Adjustable Params) started on port 50005...")
    asyncio.run(main())