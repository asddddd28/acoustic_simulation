import asyncio
import websockets
import struct
import torch
import torch.nn as nn
import numpy as np
import time
import os
import math
import matplotlib.patches as pt # [新增] 引入 patches 用于画圆
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 0. 配置区域
# ==========================================
DEBUG_SAVE_IMAGES = False       # 开关：是否保存推理热力图
DEBUG_FOLDER = "debug_frames"  # 保存文件夹名称

# [核心参数] 声源定位参数
# 1. 聚合半径：在最高峰附近多少米范围内的点被视为同一个声源
CENTROID_RADIUS = 0.05         # 单位：米 (例如 10cm)
# 2. 能量极差容忍度：纳入计算的点，其置信度必须 >= (峰值 - 容忍度)
ENERGY_RELATIVE_THRESH = 0.15 

# 初始化保存目录
if DEBUG_SAVE_IMAGES:
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    print(f"Debug images will be saved to: {os.path.abspath(DEBUG_FOLDER)}")

# ==========================================
# 1. 模型定义 (保持不变)
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
# 2. 核心算法：区域生长质心法
# ==========================================
def detect_sources_by_centroid(grid_input, global_threshold, radius_m=0.1, energy_delta=0.2):
    """
    使用迭代质心法寻找声源，并返回每个声源对应的像素掩膜
    """
    working_grid = grid_input.copy()
    rows, cols = working_grid.shape
    
    # 预计算物理坐标网格
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
            
        peak_x = X[max_r, max_c]
        peak_y = Y[max_r, max_c]
        
        # 创建掩膜
        dist_map = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
        spatial_mask = dist_map <= radius_m
        
        value_threshold = max_val - energy_delta
        energy_mask = working_grid >= value_threshold
        
        # 最终掩膜：既在距离内，能量又足够高
        final_mask = spatial_mask & energy_mask
        
        if not np.any(final_mask):
            working_grid[max_r, max_c] = 0 
            continue
            
        # 计算质心
        weights = working_grid[final_mask]
        coords_x = X[final_mask]
        coords_y = Y[final_mask]
        
        sum_w = np.sum(weights)
        if sum_w > 0:
            centroid_x = np.sum(coords_x * weights) / sum_w
            centroid_y = np.sum(coords_y * weights) / sum_w
        else:
            centroid_x, centroid_y = peak_x, peak_y
            
        detected_sources.append((centroid_x, centroid_y, max_val))
        detected_masks.append(final_mask)
        
        # 消隐
        working_grid[spatial_mask] = 0.0
        
    return detected_sources, detected_masks

def save_debug_plot(grid, sources, masks, threshold, req_id):
    """
    保存调试图片：
    1. 热力图背景
    2. 等高线 (实际计算像素)
    3. 虚线圆 (搜索半径)
    4. 红色质心点
    """
    try:
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig) 
        ax = fig.add_subplot(111)
        
        # 1. 绘制基础热力图
        im = ax.imshow(grid, origin='lower', extent=[-0.2, 0.2, -0.2, 0.2], 
                       cmap='jet', vmin=0, vmax=1.0)
        fig.colorbar(im, ax=ax, label='Confidence')
        
        # 准备网格数据用于画轮廓
        rows, cols = grid.shape
        x_axis = np.linspace(-0.2, 0.2, cols)
        y_axis = np.linspace(-0.2, 0.2, rows)
        X, Y = np.meshgrid(x_axis, y_axis)

        # 2. 绘制纳入计算范围的像素轮廓 (不规则白实线)
        for mask in masks:
            ax.contour(X, Y, mask.astype(float), levels=[0.5], colors='white', linewidths=0.8, linestyles='-')

        # 3. 绘制最终计算出的质心点 和 搜索半径
        if len(sources) > 0:
            sx = [s[0] for s in sources]
            sy = [s[1] for s in sources]
            
            # 绘制红色实心圆点 (质心)
            ax.scatter(sx, sy, c='red', marker='o', s=60, edgecolors='black', linewidths=1, zorder=10, label='Centroid')
            
            for s in sources:
                # 3.1 [新增] 绘制虚线圆 (搜索半径)
                # 以计算出的质心为圆心画圆，表示该声源的影响范围
                circle = pt.Circle((s[0], s[1]), CENTROID_RADIUS, 
                                   color='white', fill=False, linestyle='--', linewidth=0.8, alpha=0.6)
                ax.add_patch(circle)
                
                # 3.2 标注置信度数值
                ax.text(s[0]+0.01, s[1]+0.01, f"{s[2]:.2f}", color='white', fontsize=8, fontweight='bold', zorder=10)

        ax.set_title(f"Req {req_id} | Detected: {len(sources)}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        filename = os.path.join(DEBUG_FOLDER, f"req_{req_id:04d}.png")
        fig.savefig(filename)
    except Exception as e:
        print(f"    [Plot Error] {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 3. WebSocket 处理逻辑
# ==========================================
async def handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    loop = asyncio.get_running_loop()
    plot_executor = ThreadPoolExecutor(max_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AcousticResNet().to(device)
    
    try:
        weight_path = "acoustic_resnet_weights_sharp.pth"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            print("✓ Model loaded successfully")
        else:
            print(f"✗ Error: Weight file '{weight_path}' not found!")
            return
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        return

    req_count = 0
    try:
        async for message in websocket:
            try:
                req_count += 1
                start_t = time.time()
                
                # --- 数据校验 ---
                msg_len = len(message)
                if msg_len != 2049 * 4:
                    print(f"[Req {req_count}] ERROR: Invalid packet size {msg_len}")
                    continue

                # --- 解析 ---
                all_floats = np.frombuffer(message, dtype='<f4')
                threshold = all_floats[0]
                csm_real = all_floats[1:1025].reshape(32, 32, order='F')
                csm_imag = all_floats[1025:2049].reshape(32, 32, order='F')

                # --- 握手 ---
                if abs(np.trace(csm_real) - 1.0) < 0.1 and np.allclose(csm_real, np.eye(32), atol=0.1):
                    print(f"[Req {req_count}] Handshake detected -> Sending OK")
                    await websocket.send(struct.pack('<i', 0))
                    continue

                # --- 推理 ---
                inp_tensor = torch.stack([
                    torch.from_numpy(csm_real.copy()),
                    torch.from_numpy(csm_imag.copy())
                ], dim=0).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    output = model(inp_tensor).squeeze().cpu().numpy()
                
                # --- 后处理 (质心法 + 掩膜) ---
                final_sources, final_masks = detect_sources_by_centroid(
                    output, 
                    global_threshold=threshold,
                    radius_m=CENTROID_RADIUS,
                    energy_delta=ENERGY_RELATIVE_THRESH
                )
                
                total_time = time.time() - start_t
                log_msg = f"[Req {req_count}] Sources: {len(final_sources)} | {total_time*1000:.1f}ms"
                if len(final_sources) > 0: 
                    p_str = ", ".join([f"({s[0]:.3f}, {s[1]:.3f})" for s in final_sources])
                    log_msg += f" -> {p_str}"
                print(log_msg)

                # --- 异步保存图片 ---
                if DEBUG_SAVE_IMAGES:
                    loop.run_in_executor(
                        plot_executor, 
                        save_debug_plot, 
                        output.copy(), 
                        final_sources,
                        final_masks,
                        threshold, 
                        req_count
                    )

                # --- 发送响应 ---
                response = struct.pack('<i', len(final_sources))
                for s in final_sources:
                    response += struct.pack('<ff', float(s[0]), float(s[1]))
                
                try:
                    await websocket.send(response)
                except websockets.exceptions.ConnectionClosed:
                    print(f"[Req {req_count}] Client disconnected during send")
                    raise 
            
            except Exception as inner_e:
                print(f"[Req {req_count}] Processing Error: {inner_e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected (Session End)")
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up executor...")
        plot_executor.shutdown(wait=False)

async def main():
    print(f"WebSocket Server listening on 0.0.0.0:50005")
    async with websockets.serve(handler, "0.0.0.0", 50005, max_size=None, ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")