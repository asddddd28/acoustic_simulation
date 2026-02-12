---

# 声学波束成形与 AI 推理联合仿真系统 (Acoustic-AI Joint Simulation)

本项目是一个集成了传统信号处理（波束成形）与深度学习（ResNet 定位）的实时声场仿真平台。通过 MATLAB 进行物理声场模拟与 UI 交互，利用 Python 进行高性能 AI 推理，二者通过 WebSocket 协议实现低延迟数据同步。

## 1. 项目架构

系统由三个主要模块组成：

1. **仿真前端 (MATLAB)**：负责麦克风阵列模拟、声源移动计算、传统波束成形热力图渲染及 WebSocket 客户端实现。
2. **推理后端 (Python)**：基于 PyTorch 实现的实时推理服务器，接收 CSM（跨谱矩阵）数据并返回声源坐标。
3. **训练模块 (Python)**：用于离线训练声学卷积神经网络，生成稳健的定位模型。

---

## 2. 目录结构与文件说明

### 核心程序

* **`AcousticSimRealFluentModelApp.m`**：**主控程序**。基于 MATLAB App Designer 开发的实时仿真界面，具有极高的流畅度，支持手动移动声源（WASDQE）与 AI 参数实时微调。
* **`inference_server_v3.py`**：**推理服务器**。基于 `websockets` 库的高性能后端，支持 GPU 加速推理、动态参数调整及 Debug 图像保存。
* **`fibonacci_32_user.xml`**：麦克风阵列配置，定义了 32 个通道的斐波那契螺旋布局。

### 训练与模型

* **`train_acoustic_net_v2.py`**：声学定位网络训练脚本，支持数据增强与鲁棒性训练。
* **`acoustic_resnet_robust_best.pth`**：当前性能最优的预训练模型权重。
* **`acoustic_resnet.onnx`**：导出的 ONNX 格式模型，用于跨平台部署。

### 辅助与历史版本

* `debug_frames/`：存放 Python 端生成的 AI 预测热力图与掩码结果（需开启 Debug Plot）。
* `AcousticSimRealApp.m` / `inference_server_v2.py`：旧版本参考程序。

---

## 3. 环境要求

### MATLAB

* 版本：R2022b 或更高版本。
* 工具箱：
* System Optimization Toolbox (可选)
* Parallel Computing Toolbox (若需 GPU 加速计算波束成形)



### Python

* Python 3.8+
* 核心依赖：
```bash
pip install torch torchvision numpy websockets matplotlib

```


* 建议配备支持 CUDA 的 NVIDIA GPU 以实现极致推理速度。

---

## 4. 快速开始

### 第一步：启动 AI 推理服务器

在终端执行以下命令，启动后端监听（默认端口 50005）：

```bash
python inference_server_v3.py

```

当看到 `✓ Robust model loaded` 和 `Server started` 时，说明后端就绪。

### 第二步：运行 MATLAB 仿真

1. 在 MATLAB 中打开 `AcousticSimRealFluentModelApp.m` 并运行。
2. 在左侧面板 **"4. AI 推理辅助"** 中，将 **Enable** 切换至 **On**。此时 MATLAB 会建立 WebSocket 连接。
3. 将 **实时仿真模式** 切换至 **On**。

### 第三步：交互与操控

* **移动声源**：在列表框选择声源后，使用 `W/A/S/D`（平移）和 `Q/E`（深度控制）实时移动声源。
* **参数微调**：
* **Thresh**：调整 AI 判定的置信度阈值。
* **Radius**：调整质心聚合半径。
* **Energy**：调整局部能量容忍度。


* **结果观察**：洋红色菱形 (`◇`) 代表 AI 预测位置，白色五角星 (`★`) 代表真实物理位置。

---

## 5. 数据协议说明 (WebSocket)

MATLAB 与 Python 之间每帧交换的数据量固定为 **8208 字节**：

1. **控制参数 (16 字节)**：4 个 `float32`（阈值、半径、能量容忍度、Debug 标志）。
2. **CSM 实部 (4096 字节)**： 的单精度浮点数矩阵。
3. **CSM 虚部 (4096 字节)**： 的单精度浮点数矩阵。

---

## 6. 开发建议

* **性能优化**：若实时模式出现卡顿，请在 MATLAB 界面中适当调大频率步长或减小扫描分辨率。
* **模型更新**：若需更换阵列布局，请重新运行 `train_acoustic_net_v2.py` 训练对应布局的模型，并更新服务器中的权重路径。

---

*Created by Gemini for Acoustic Simulation Project | 2026.02.12*
