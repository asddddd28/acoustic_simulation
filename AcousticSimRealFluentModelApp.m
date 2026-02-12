classdef AcousticSimRealFluentModelApp < handle
    % ACOUSTICSIMREALFLUENTMODELAPP 
    % 修复版 V4: 解决 DiscreteKnob 类型属性报错、uibutton语法及变量名冲突
    
    properties
        % UI 组件
        UIFigure        matlab.ui.Figure
        GridLayout      matlab.ui.container.GridLayout
        LeftPanel       matlab.ui.container.Panel
        RightTabs       matlab.ui.container.TabGroup
        
        % 常规视图 Tab
        Tab3D           matlab.ui.container.Tab
        TabBF           matlab.ui.container.Tab
        Ax3D            matlab.graphics.axis.Axes
        AxBF_Map        matlab.graphics.axis.Axes
        AxBF_Array      matlab.graphics.axis.Axes
        
        % 实时视图 Tab
        TabRealTime     matlab.ui.container.Tab
        RTContainerGrid matlab.ui.container.GridLayout 
        RTPlotGrid      matlab.ui.container.GridLayout 
        Ax3D_RT         matlab.graphics.axis.Axes      
        AxBF_RT         matlab.graphics.axis.Axes      
        LayoutSwitch    matlab.ui.control.Switch       
        
        % 控件
        SourceListBox   matlab.ui.control.ListBox
        StatusLabel     matlab.ui.control.Label
        RealTimeSwitch  matlab.ui.control.Switch
        
        % 输入控件
        EditX, EditY, EditZ
        EditFreq, EditZFixed, EditZStart, EditZEnd, EditZStep
        EditSNR         matlab.ui.control.NumericEditField 
        ModeSwitch      matlab.ui.control.Switch
        ModelDropDown   matlab.ui.control.DropDown
        ShowStarsCheck  matlab.ui.control.CheckBox
        
        % 算法控件
        AlgoDropDown    matlab.ui.control.DropDown
        NoiseTypeDropDown matlab.ui.control.DropDown
        EditMusicN      matlab.ui.control.NumericEditField
        EditFuncGamma   matlab.ui.control.NumericEditField
        
        % === AI 推理控件 (Updated) ===
        PnlDL           matlab.ui.container.Panel
        DLSwitch        matlab.ui.control.Switch
        
        DLThreshKnob     % 离散或连续均可
        DLThreshLabel   matlab.ui.control.Label
        
        DLRadiusKnob     % 新增: 聚合半径
        DLRadiusLabel   matlab.ui.control.Label
        
        DLEnergyKnob     % 新增: 能量容忍度
        DLEnergyLabel   matlab.ui.control.Label
        
        DLDebugCheck    matlab.ui.control.CheckBox % 新增: Debug图片输出
        
        TcpClient       % TCP 通信对象
        DLDetectedSources double = [] % 存储 AI 预测的坐标 [N x 2]
        hDLMarkers      % AI 结果绘图句柄
        
        % 仿真数据
        MicCoords       double % 3xN 矩阵
        Sources         double % Nx3 矩阵 (x,y,z)
        SelectedSource  double = [] 
        
        % 绘图句柄缓存
        hMapImage       
        hRealTimeSrc    
        hRealTimeMics   
        h3DSources      
        h3DSelected     
        
        % 视图状态
        ViewAzim        double = 45
        ViewElev        double = 20
        
        % 系统状态
        SimTimer        timer
        IsRealTime      logical = false
        HasGPU          logical = false 
        
        % 常量
        c               double = 343; 
        SampleRate      double = 51200;
        
        % 缓存网格 
        CachedGridX
        CachedGridY
        CachedRes       double = 0 
    end
    
    methods
        function obj = AcousticSimRealFluentModelApp()
            % 构造函数
            obj.checkGPU();
            obj.initData();
            obj.createUI();
            obj.initTimer();
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D); 
        end
        
        function delete(obj)
            % 析构函数：清理定时器和 TCP 连接
            if ~isempty(obj.SimTimer) && isvalid(obj.SimTimer)
                stop(obj.SimTimer);
                delete(obj.SimTimer);
            end
            if ~isempty(obj.TcpClient)
                delete(obj.TcpClient);
            end
            delete(obj.UIFigure);
        end
        
        function checkGPU(obj)
            try
                count = gpuDeviceCount;
                if count > 0
                    obj.HasGPU = true;
                    dev = gpuDevice;
                    fprintf('检测到 GPU: %s (显存: %.1f GB)\n', dev.Name, dev.TotalMemory/1e9);
                else
                    obj.HasGPU = false;
                    fprintf('未检测到兼容的 GPU，将使用 CPU 向量化加速。\n');
                end
            catch
                obj.HasGPU = false;
            end
        end
        
        function initData(obj)
            xmlFile = 'fibonacci_32_user.xml';
            if exist(xmlFile, 'file') == 2
                obj.loadMicFromXML(xmlFile);
            else
                obj.loadDefaultArray();
            end
            obj.Sources = [];
        end
        
        function initTimer(obj)
            % 提速到 20 FPS (Period 0.05)
            obj.SimTimer = timer('ExecutionMode', 'fixedRate', ...
                'Period', 0.05, ... 
                'TimerFcn', @obj.onRealTimeStep);
        end

        function loadMicFromXML(obj, filename)
            try
                xDoc = xmlread(filename);
                positions = xDoc.getElementsByTagName('pos');
                n_mics = positions.getLength();
                
                x = zeros(1, n_mics);
                y = zeros(1, n_mics);
                z = zeros(1, n_mics);
                
                for k = 0:n_mics-1
                    item = positions.item(k);
                    x(k+1) = str2double(char(item.getAttribute('x')));
                    y(k+1) = str2double(char(item.getAttribute('y')));
                    z(k+1) = str2double(char(item.getAttribute('z')));
                end
                
                obj.MicCoords = [x; y; z];
            catch
                obj.loadDefaultArray();
            end
        end

        function loadDefaultArray(obj)
            n_mics = 32;
            r = linspace(0.01, 0.2, n_mics);
            phi = linspace(0, 2*pi*3, n_mics);
            x = r .* cos(phi);
            y = r .* sin(phi);
            z = zeros(1, n_mics);
            obj.MicCoords = [x; y; z]; 
        end
        
        function createUI(obj)
            titleStr = '声场生成与波束成形仿真系统 (GPU + AI Hybrid V2)';
            if obj.HasGPU, titleStr = [titleStr ' - GPU ON']; end
            
            obj.UIFigure = uifigure('Name', titleStr, ...
                'Position', [100, 100, 1300, 920]); % 稍微增高
            
            obj.UIFigure.WindowKeyPressFcn = @obj.onKeyPress;
            
            obj.GridLayout = uigridlayout(obj.UIFigure, [1, 2]);
            obj.GridLayout.ColumnWidth = {380, '1x'}; % 左侧稍微加宽以容纳更多控件
            
            % === 左侧控制面板 ===
            obj.LeftPanel = uipanel(obj.GridLayout, 'Title', '控制面板');
            obj.LeftPanel.Layout.Row = 1; obj.LeftPanel.Layout.Column = 1;
            
            vbox = uigridlayout(obj.LeftPanel, [9, 1]); 
            vbox.RowHeight = {'fit', 'fit', 'fit', 'fit', 180, 'fit', 'fit', '1x', 'fit'};
            vbox.Scrollable = 'on';
            
            % 1. 声源管理
            pnlSrc = uipanel(vbox, 'Title', '1. 声源管理 (WASD+QE移动)');
            srcGrid = uigridlayout(pnlSrc, [2, 6]);
            srcGrid.RowHeight = {'fit', 'fit'};
            uilabel(srcGrid, 'Text', 'X:');
            obj.EditX = uieditfield(srcGrid, 'numeric', 'Value', 0.0);
            uilabel(srcGrid, 'Text', 'Y:');
            obj.EditY = uieditfield(srcGrid, 'numeric', 'Value', 0.0);
            uilabel(srcGrid, 'Text', 'Z:');
            obj.EditZ = uieditfield(srcGrid, 'numeric', 'Value', -0.3);
            uibutton('Parent', srcGrid, 'Text', '添加', 'ButtonPushedFcn', @obj.onAddSource);
            uibutton('Parent', srcGrid, 'Text', '删除', 'ButtonPushedFcn', @obj.onRemoveSource);
            uibutton('Parent', srcGrid, 'Text', '清空', 'ButtonPushedFcn', @obj.onClearSources);
            
            obj.SourceListBox = uilistbox(vbox, 'ValueChangedFcn', @obj.onSourceSelect);
            obj.SourceListBox.Layout.Row = 2;
            
            % 2. 扫描设置
            pnlScan = uipanel(vbox, 'Title', '2. 扫描设置');
            pnlScan.Layout.Row = 3;
            scanGrid = uigridlayout(pnlScan, [6, 2]);
            scanGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit'};
            uilabel(scanGrid, 'Text', '模型类型:');
            obj.ModelDropDown = uidropdown(scanGrid, 'Items', {'近场 (True Level)', '远场 (Plane Wave)'}, ...
                'ItemsData', {'near', 'far'}, 'Value', 'near');
            obj.ShowStarsCheck = uicheckbox(scanGrid, 'Text', '显示真实声源(★)', 'Value', true);
            obj.ShowStarsCheck.Layout.Column = [1 2];
            uilabel(scanGrid, 'Text', '扫描模式:');
            obj.ModeSwitch = uiswitch(scanGrid, 'Items', {'固定深度', '范围(MIP)'}, 'ValueChangedFcn', @obj.onModeChanged);
            uilabel(scanGrid, 'Text', '固定深度 Z(m):');
            obj.EditZFixed = uieditfield(scanGrid, 'numeric', 'Value', -0.3);
            uilabel(scanGrid, 'Text', '范围 Z(s/e/step):');
            rangeBox = uigridlayout(scanGrid, [1, 3]); rangeBox.Padding = [0 0 0 0]; rangeBox.Layout.Column = 2;
            obj.EditZStart = uieditfield(rangeBox, 'numeric', 'Value', -0.1, 'Enable', 'off');
            obj.EditZEnd = uieditfield(rangeBox, 'numeric', 'Value', -0.5, 'Enable', 'off');
            obj.EditZStep = uieditfield(rangeBox, 'numeric', 'Value', 0.02, 'Enable', 'off');
            btnPreset = uibutton('Parent', scanGrid, 'Text', '从当前声源预设参数', 'ButtonPushedFcn', @obj.onPresetParams);
            btnPreset.Layout.Column = [1 2];
            
            % 3. 算法参数
            pnlAlgo = uipanel(vbox, 'Title', '3. 算法运行');
            pnlAlgo.Layout.Row = 4;
            algoGrid = uigridlayout(pnlAlgo, [6, 2]); 
            algoGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit'};
            uilabel(algoGrid, 'Text', '中心频率 (Hz):');
            obj.EditFreq = uieditfield(algoGrid, 'numeric', 'Value', 8000);
            uilabel(algoGrid, 'Text', '信噪比 SNR (dB):');
            obj.EditSNR = uieditfield(algoGrid, 'numeric', 'Value', 30, 'Limits', [-20 100]);
            uilabel(algoGrid, 'Text', '噪声模型:');
            obj.NoiseTypeDropDown = uidropdown(algoGrid, 'Items', {'传感器白噪声 (White)', '扩散场噪声 (Diffuse)', '湍流噪声 (Turbulence)'}, 'Value', '传感器白噪声 (White)');
            uilabel(algoGrid, 'Text', '算法类型:');
            obj.AlgoDropDown = uidropdown(algoGrid, 'Items', {'Conventional', 'MUSIC', 'Functional'}, 'Value', 'Conventional');
            paramGrid = uigridlayout(algoGrid, [1, 4]); paramGrid.Layout.Column = [1 2]; paramGrid.Padding = [0 0 0 0];
            uilabel(paramGrid, 'Text', 'MUSIC源数:');
            obj.EditMusicN = uieditfield(paramGrid, 'numeric', 'Value', 1, 'Limits', [1 100]);
            uilabel(paramGrid, 'Text', 'Gamma:');
            obj.EditFuncGamma = uieditfield(paramGrid, 'numeric', 'Value', 10, 'Limits', [1 1000]);
            btnRun = uibutton('Parent', algoGrid, 'Text', '单次运行', 'BackgroundColor', [0.3, 0.6, 1.0], 'FontWeight', 'bold', 'ButtonPushedFcn', @obj.runBeamforming);
            btnRun.Layout.Column = [1 2];
            
            % === 4. 深度学习推理 (增强版) ===
            obj.PnlDL = uipanel(vbox, 'Title', '4. AI 推理辅助 (参数微调)');
            obj.PnlDL.Layout.Row = 5; 
            
            % 主网格：分为上下两层
            % Row 1: 开关与设置 (高度 fit)
            % Row 2: 旋钮区域 (高度 1x，自动撑大)
            mainDLGrid = uigridlayout(obj.PnlDL, [2, 1]);
            mainDLGrid.RowHeight = {'fit', '1x'}; 
            mainDLGrid.Padding = [10 5 10 5];
            mainDLGrid.RowSpacing = 5;
            
            % --- 第一行：开关区域 ---
            topRowGrid = uigridlayout(mainDLGrid, [1, 3]);
            topRowGrid.Layout.Row = 1;
            topRowGrid.ColumnWidth = {'fit', 'fit', '1x'};
            topRowGrid.Padding = [0 0 0 0];
            
            uilabel(topRowGrid, 'Text', 'Enable:', 'FontWeight', 'bold');
            obj.DLSwitch = uiswitch(topRowGrid, 'slider', 'Items', {'Off', 'On'}, ...
                'ValueChangedFcn', @obj.onDLToggle);
            
            obj.DLDebugCheck = uicheckbox(topRowGrid, 'Text', 'Debug Plot');
            obj.DLDebugCheck.Layout.Column = 3; % 靠右对齐
            
            % --- 第二行：三个大旋钮区域 ---
            knobContainer = uigridlayout(mainDLGrid, [1, 3]);
            knobContainer.Layout.Row = 2;
            knobContainer.ColumnSpacing = 20; % 增加间距防止拥挤
            knobContainer.Padding = [0 10 0 0];
            
            % 1. 阈值旋钮 (Threshold)
            k1Grid = uigridlayout(knobContainer, [2, 1]);
            k1Grid.RowHeight = {'1x', 'fit'}; k1Grid.Padding = 0;
            obj.DLThreshKnob = uiknob(k1Grid, 'discrete', ...
                'Items', {'0.1', '0.3', '0.5', '0.7', '0.9'}, ...
                'ItemsData', [0.1, 0.3, 0.5, 0.7, 0.9], ...
                'Value', 0.5, ...
                'ValueChangedFcn', @(s,e) set(obj.DLThreshLabel, 'Text', sprintf('Thresh: %.1f', s.Value)));
            obj.DLThreshLabel = uilabel(k1Grid, 'Text', 'Thresh: 0.5', 'HorizontalAlignment', 'center');
            
            % 2. 半径旋钮 (Radius)
            k2Grid = uigridlayout(knobContainer, [2, 1]);
            k2Grid.RowHeight = {'1x', 'fit'}; k2Grid.Padding = 0;
            obj.DLRadiusKnob = uiknob(k2Grid, 'continuous', ...
                'Limits', [0.02 0.22], 'Value', 0.05, ...
                'ValueChangedFcn', @(s,e) set(obj.DLRadiusLabel, 'Text', sprintf('Rad: %.2fm', s.Value)));
            obj.DLRadiusLabel = uilabel(k2Grid, 'Text', 'Rad: 0.05m', 'HorizontalAlignment', 'center');
            
            % 3. 能量旋钮 (Energy)
            k3Grid = uigridlayout(knobContainer, [2, 1]);
            k3Grid.RowHeight = {'1x', 'fit'}; k3Grid.Padding = 0;
            obj.DLEnergyKnob = uiknob(k3Grid, 'continuous', ...
                'Limits', [0.05 0.45], 'Value', 0.15, ...
                'ValueChangedFcn', @(s,e) set(obj.DLEnergyLabel, 'Text', sprintf('Egy: %.2f', s.Value)));
            obj.DLEnergyLabel = uilabel(k3Grid, 'Text', 'Egy: 0.15', 'HorizontalAlignment', 'center');

            % 实时开关区域
            rtPanel = uipanel(vbox, 'BorderType', 'none'); rtPanel.Layout.Row = 6;
            rtGrid = uigridlayout(rtPanel, [1, 2]); rtGrid.Padding = [0 5 0 5];
            uilabel(rtGrid, 'Text', '实时仿真模式:', 'FontWeight', 'bold', 'FontColor', [0.8 0.2 0]);
            obj.RealTimeSwitch = uiswitch(rtGrid, 'Items', {'Off', 'On'}, 'ValueChangedFcn', @obj.onRealTimeToggle);
            
            accelType = 'CPU Vector'; if obj.HasGPU, accelType = 'GPU Accelerated'; end
            obj.StatusLabel = uilabel(vbox, 'Text', sprintf('就绪 [%s] | 提示: WASD移动声源', accelType));
            obj.StatusLabel.Layout.Row = 9; obj.StatusLabel.FontColor = [0.4 0.4 0.4];
            
            % === 右侧绘图区 ===
            obj.RightTabs = uitabgroup(obj.GridLayout);
            obj.RightTabs.Layout.Row = 1; obj.RightTabs.Layout.Column = 2;
            
            obj.Tab3D = uitab(obj.RightTabs, 'Title', '3D 设置视图');
            gl3d = uigridlayout(obj.Tab3D, [1,1]);
            obj.Ax3D = uiaxes(gl3d);
            xlabel(obj.Ax3D, 'X (m)'); ylabel(obj.Ax3D, 'Y (m)'); zlabel(obj.Ax3D, 'Z (m)');
            grid(obj.Ax3D, 'on'); title(obj.Ax3D, '声源位置配置 (Setup View)');
            view(obj.Ax3D, obj.ViewAzim, obj.ViewElev);
            
            obj.TabBF = uitab(obj.RightTabs, 'Title', '波束成形热力图');
            glbf = uigridlayout(obj.TabBF, [2, 1]); glbf.RowHeight = {'2x', '1x'};
            obj.AxBF_Map = uiaxes(glbf); title(obj.AxBF_Map, '波束成形结果 (Result View)'); axis(obj.AxBF_Map, 'equal');
            obj.AxBF_Array = uiaxes(glbf); title(obj.AxBF_Array, '麦克风布局');
            
            obj.TabRealTime = uitab(obj.RightTabs, 'Title', '实时仿真 (Real-time)');
            obj.RTContainerGrid = uigridlayout(obj.TabRealTime, [2, 1]); obj.RTContainerGrid.RowHeight = {'fit', '1x'};
            toolbarGrid = uigridlayout(obj.RTContainerGrid, [1, 3]); toolbarGrid.ColumnWidth = {'fit', 'fit', '1x'};
            toolbarGrid.Layout.Row = 1;
            uilabel(toolbarGrid, 'Text', '界面布局:');
            obj.LayoutSwitch = uiswitch(toolbarGrid, 'Items', {'左右分屏', '上下分屏'}, 'ValueChangedFcn', @obj.onLayoutChange);
            obj.RTPlotGrid = uigridlayout(obj.RTContainerGrid, [1, 2]); 
            obj.RTPlotGrid.Layout.Row = 2; obj.RTPlotGrid.ColumnWidth = {'1x', '1x'};
            obj.Ax3D_RT = uiaxes(obj.RTPlotGrid);
            obj.Ax3D_RT.Layout.Row = 1; obj.Ax3D_RT.Layout.Column = 1;
            title(obj.Ax3D_RT, '实时空间视图 (可交互)'); xlabel(obj.Ax3D_RT, 'X'); ylabel(obj.Ax3D_RT, 'Y'); zlabel(obj.Ax3D_RT, 'Z'); grid(obj.Ax3D_RT, 'on');
            obj.AxBF_RT = uiaxes(obj.RTPlotGrid);
            obj.AxBF_RT.Layout.Row = 1; obj.AxBF_RT.Layout.Column = 2;
            title(obj.AxBF_RT, '实时波束成形'); axis(obj.AxBF_RT, 'equal');
        end

         % === WebSocket 协议核心实现 ===
        function ws_handshake(obj)
            % 1. 生成随机 Key
            keyChars = ['A':'Z', 'a':'z', '0':'9', '+', '/'];
            randKey = keyChars(randi(length(keyChars), 1, 22)); % 简化的 Base64 长度模拟
            key = [randKey '=='];
            
            % 2. 发送 HTTP Upgrade 请求
            req = sprintf('GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n', key);
            write(obj.TcpClient, uint8(req));
            
            % 3. 读取响应 (简化处理：读取直到两个CRLF)
            % 实际应该检查 Sec-WebSocket-Accept，但为了演示这里跳过验证
            resp = '';
            tic;
            while ~contains(resp, [char(13) char(10) char(13) char(10)]) && toc < 3
                if obj.TcpClient.NumBytesAvailable > 0
                    chunk = read(obj.TcpClient, obj.TcpClient.NumBytesAvailable, 'char');
                    resp = [resp char(chunk)];
                end
                pause(0.01);
            end
            
            if contains(resp, '101 Switching Protocols')
                fprintf('[WS] 握手成功！\n');
            else
                error('WS握手失败: %s', resp);
            end
        end
        
        function ws_send(obj, data_float)
            % data_float: float32 column vector
            
            % 1. 转换为字节
            payload = typecast(single(data_float(:)), 'uint8'); 
            len = length(payload);
            
            % 2. 构造帧头 (FIN=1, Opcode=2 Binary)
            frame = uint8([]);
            frame(1) = 130; 
            
            % 3. 长度字段 + Mask bit (0x80)
            if len <= 125
                frame(2) = bitor(128, len);
            elseif len <= 65535
                frame(2) = bitor(128, 126);
                frame(3) = bitshift(len, -8);
                frame(4) = bitand(len, 255);
            else
                error('Payload too large');
            end
            
            % 4. 生成 Masking Key
            maskKey = randi([0 255], 1, 4, 'uint8');
            frame = [frame, maskKey];
            
            % 5. 掩码处理 (XOR)
            expandedMask = repmat(maskKey, 1, ceil(len/4));
            expandedMask = expandedMask(1:len);
            
            % 确保维度一致 (Explicit Transpose)
            maskedPayload = bitxor(payload(:)', expandedMask(:)');
            
            % 6. 发送
            write(obj.TcpClient, [frame, maskedPayload]);
            fprintf('[WS] Sent %d bytes payload\n', len); % 调试用
        end
        
        function peaks = ws_recv(obj)
            % 接收 WebSocket Frame (Server -> Client 无掩码)
            peaks = [];
            
            % 1. 读取头 2 字节
            if obj.TcpClient.NumBytesAvailable < 2, return; end
            header = read(obj.TcpClient, 2, 'uint8');
            
            % Opcode 检查 (0x82=Binary)
            if header(1) ~= 130, return; end
            
            % 长度解析
            len = double(bitand(header(2), 127));
            if len == 126
                % 读取接下来的 2 字节长度
                while obj.TcpClient.NumBytesAvailable < 2, pause(0.001); end
                lenBytes = read(obj.TcpClient, 2, 'uint8');
                len = double(lenBytes(1))*256 + double(lenBytes(2));
            end
            
            % 2. 读取 Payload
            while obj.TcpClient.NumBytesAvailable < len, pause(0.001); end
            payload = read(obj.TcpClient, len, 'uint8');
            
            % 3. 解析业务数据 (int count, float x, float y...)
            if isempty(payload), return; end
            
            num_peaks = typecast(payload(1:4), 'int32');
            if num_peaks > 0
                raw_coords = typecast(payload(5:end), 'single');
                peaks = reshape(raw_coords, 2, num_peaks).';
            end
        end

        % === AI 开关回调 ===
        % === 回调函数变更 ===
        function onFreqChanged(obj,s,~), if strcmp(obj.DLSwitch.Value,'On')&&abs(s.Value-8000)>1, uialert(obj.UIFigure,'AI需8000Hz','警报'); end, end

        function onDLToggle(obj, src, ~)
            if strcmp(src.Value, 'On')
                if abs(obj.EditFreq.Value-8000)>1, uialert(obj.UIFigure,'请使用8000Hz','提示'); end
                try
                    fprintf('[WS] 连接中...\n');
                    obj.TcpClient = tcpclient('127.0.0.1', 50005, 'Timeout', 5);
                    
                    % *** 执行 WebSocket 握手 ***
                    obj.ws_handshake();
                    
                    % 发送握手包 (单位矩阵)
                    hs_csm = eye(32,'single')/32;
                    data = [single(0.5);single(0.05);single(0.15);single(0.0); real(hs_csm(:)); imag(hs_csm(:))];
                    
                    obj.ws_send(data);
                    
                    % 等待响应
                    pause(0.1);
                    obj.ws_recv(); % 丢弃握手响应
                    
                    obj.StatusLabel.Text = '✓ AI (WebSocket) 已连接';
                    obj.StatusLabel.FontColor = [0 0.6 0];
                catch ME
                    fprintf('[WS Error] %s\n', ME.message);
                    src.Value = 'Off'; if ~isempty(obj.TcpClient), delete(obj.TcpClient); end
                    uialert(obj.UIFigure, ME.message, 'WS连接失败');
                end
            else
                if ~isempty(obj.TcpClient), delete(obj.TcpClient); end
                obj.StatusLabel.Text = 'AI 已关闭'; obj.StatusLabel.FontColor = [0.4 0.4 0.4];
                obj.DLDetectedSources = [];
            end
        end
        
        % === 布局切换逻辑 ===
        function onLayoutChange(obj, src, ~)
            val = src.Value;
            if strcmp(val, '左右分屏')
                obj.RTPlotGrid.RowHeight = {'1x'};
                obj.RTPlotGrid.ColumnWidth = {'1x', '1x'};
                obj.Ax3D_RT.Layout.Row = 1; obj.Ax3D_RT.Layout.Column = 1;
                obj.AxBF_RT.Layout.Row = 1; obj.AxBF_RT.Layout.Column = 2;
            else
                obj.RTPlotGrid.RowHeight = {'1x', '1x'};
                obj.RTPlotGrid.ColumnWidth = {'1x'};
                obj.Ax3D_RT.Layout.Row = 1; obj.Ax3D_RT.Layout.Column = 1;
                obj.AxBF_RT.Layout.Row = 2; obj.AxBF_RT.Layout.Column = 1;
            end
        end
        
        % === 实时模式逻辑 ===
        function onRealTimeToggle(obj, src, ~)
            if strcmp(src.Value, 'On')
                obj.IsRealTime = true;
                obj.RightTabs.SelectedTab = obj.TabRealTime;
                
                % 清空句柄缓存
                obj.hMapImage = [];
                obj.hRealTimeSrc = [];
                obj.hRealTimeMics = [];
                obj.hDLMarkers = []; % 清空 AI 标记句柄
                
                [az, el] = view(obj.Ax3D);
                view(obj.Ax3D_RT, az, el);
                obj.plot3DScene(obj.Ax3D_RT);
                
                start(obj.SimTimer);
                
                modeStr = 'CPU Vector';
                if obj.HasGPU, modeStr = 'GPU Accelerated'; end
                obj.StatusLabel.Text = sprintf('实时仿真运行中 (%s)... WASD移动', modeStr);
            else
                obj.IsRealTime = false;
                stop(obj.SimTimer);
                obj.RightTabs.SelectedTab = obj.TabBF;
                obj.StatusLabel.Text = '实时仿真已停止';
            end
        end
        
        function onRealTimeStep(obj, ~, ~)
            try
                baseSNR = obj.EditSNR.Value;
                currentSNR = baseSNR + 2 * randn(); 
                
                % 调用核心计算 (包含 AI 通信)
                [map, max_val, GX, GY] = obj.calculateBeamformingMap(currentSNR);
                
                % 刷新实时热力图
                obj.plotResultsRT(GX, GY, map, max_val);
                
            catch ME
                fprintf('[WS Error] %s\n', ME.message);
                stop(obj.SimTimer);
                obj.RealTimeSwitch.Value = 'Off';
                obj.onRealTimeToggle(obj.RealTimeSwitch, []);
                uialert(obj.UIFigure, ME.message, '实时仿真出错');
            end
        end
        
        % === 交互回调 ===
        function onKeyPress(obj, ~, event)
            if isempty(obj.SelectedSource) || size(obj.Sources, 1) < obj.SelectedSource
                return; 
            end
            
            step = 0.01;
            dx = 0; dy = 0; dz = 0;
            switch event.Key
                case {'w', 'W'}, dy = step;
                case {'s', 'S'}, dy = -step;
                case {'a', 'A'}, dx = -step;
                case {'d', 'D'}, dx = step;
                case {'q', 'Q'}, dz = -step;
                case {'e', 'E'}, dz = step;
                case 'delete'
                    obj.onRemoveSource([],[]);
                    return;
                otherwise
                    return;
            end
            
            obj.Sources(obj.SelectedSource, :) = obj.Sources(obj.SelectedSource, :) + [dx, dy, dz];
            s = obj.Sources(obj.SelectedSource, :);
            obj.EditX.Value = s(1);
            obj.EditY.Value = s(2);
            obj.EditZ.Value = s(3);
            
            obj.updateSourceList();
            if obj.IsRealTime
                obj.plot3DScene(obj.Ax3D_RT);
            else
                obj.plot3DScene(obj.Ax3D);
            end
        end

        function plot3DScene(obj, ax)
            % 3D 场景绘制函数 - 修复空数组索引问题
            isRT = (ax == obj.Ax3D_RT);
            if ~isRT
                cla(ax);
                hold(ax, 'on');
            else
                hold(ax, 'on');
            end
            
            mx = obj.MicCoords(1,:); my = obj.MicCoords(2,:); mz = obj.MicCoords(3,:);
            
            % 绘制麦克风阵列
            if ~isRT
                scatter3(ax, mx, my, mz, 50, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
                theta = 0:0.1:2*pi;
                max_r = max(sqrt(mx.^2 + my.^2)) * 1.1;
                fill3(ax, max_r*cos(theta), max_r*sin(theta), zeros(size(theta)), ...
                    [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'k');
            elseif isempty(obj.hRealTimeMics) || ~isvalid(obj.hRealTimeMics)
                obj.hRealTimeMics = scatter3(ax, mx, my, mz, 50, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
                theta = 0:0.1:2*pi;
                max_r = max(sqrt(mx.^2 + my.^2)) * 1.1;
                fill3(ax, max_r*cos(theta), max_r*sin(theta), zeros(size(theta)), ...
                    [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'k');
            end
            
            % 获取声源数量
            nSrc = size(obj.Sources, 1);
            
            if ~isRT
                % === 常规视图模式 ===
                if nSrc > 0
                    % 【修复点】：仅在有声源时提取坐标
                    sx = obj.Sources(:,1); sy = obj.Sources(:,2); sz = obj.Sources(:,3);
                    for i = 1:nSrc
                        plot3(ax, [sx(i) sx(i)], [sy(i) sy(i)], [sz(i) 0], 'r--', 'LineWidth', 1);
                    end
                    colors = repmat([1 0 0], nSrc, 1);
                    sizes = repmat(100, nSrc, 1);
                    if ~isempty(obj.SelectedSource) && obj.SelectedSource <= nSrc
                        colors(obj.SelectedSource, :) = [1 0.6 0]; 
                        sizes(obj.SelectedSource) = 200;
                    end
                    scatter3(ax, sx, sy, sz, sizes, colors, 'filled', 'p'); 
                end
                axis(ax, 'equal'); zlim(ax, [-0.6 0.1]); hold(ax, 'off');
            else
                % === 实时视图模式 ===
                if isempty(obj.h3DSources) || ~isvalid(obj.h3DSources) || (nSrc > 0 && length(obj.h3DSources.XData) ~= nSrc) || (nSrc == 0 && ~isempty(obj.h3DSources.XData))
                    % 重建声源点句柄
                    delete(findobj(ax, 'Type', 'Scatter', 'Marker', 'p'));
                    delete(findobj(ax, 'Type', 'Line', 'LineStyle', '--')); 
                    if nSrc > 0
                        sx = obj.Sources(:,1); sy = obj.Sources(:,2); sz = obj.Sources(:,3);
                        colors = repmat([1 0 0], nSrc, 1);
                        sizes = repmat(100, nSrc, 1);
                        if ~isempty(obj.SelectedSource) && obj.SelectedSource <= nSrc
                            colors(obj.SelectedSource, :) = [1 0.6 0]; 
                            sizes(obj.SelectedSource) = 200;
                        end
                        obj.h3DSources = scatter3(ax, sx, sy, sz, sizes, colors, 'filled', 'p');
                    end
                else
                    % 更新现有句柄数据
                    if nSrc > 0
                        sx = obj.Sources(:,1); sy = obj.Sources(:,2); sz = obj.Sources(:,3);
                        set(obj.h3DSources, 'XData', sx, 'YData', sy, 'ZData', sz);
                        colors = repmat([1 0 0], nSrc, 1);
                        sizes = repmat(100, nSrc, 1);
                        if ~isempty(obj.SelectedSource) && obj.SelectedSource <= nSrc
                            colors(obj.SelectedSource, :) = [1 0.6 0]; 
                            sizes(obj.SelectedSource) = 200;
                        end
                        set(obj.h3DSources, 'CData', colors, 'SizeData', sizes);
                    end
                end
                axis(ax, 'equal'); zlim(ax, [-0.6 0.1]); 
            end
        end
        
        function plotResultsRT(obj, GridX, GridY, map_dB, max_dB)
            ax = obj.AxBF_RT;
            algo = obj.AlgoDropDown.Value;
            if strcmp(algo, 'MUSIC'), dr = 40; 
            elseif strcmp(algo, 'Functional'), dr = 30; 
            else, dr = 15; 
            end
            
            clims = [max_dB - dr, max_dB];
            
            % 绘制/更新热力图
            if isempty(obj.hMapImage) || ~isvalid(obj.hMapImage)
                cla(ax); hold(ax, 'on');
                obj.hMapImage = imagesc(ax, GridX(1,:), GridY(:,1), map_dB);
                colormap(ax, 'jet'); axis(ax, 'xy'); axis(ax, 'equal');
                obj.hMapImage.CDataMapping = 'scaled';
                ax.CLim = clims;
                
                if obj.ShowStarsCheck.Value && ~isempty(obj.Sources)
                   sx = obj.Sources(:,1); sy = obj.Sources(:,2);
                   obj.hRealTimeSrc = plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
                end
                hold(ax, 'off');
            else
                set(obj.hMapImage, 'CData', map_dB);
                ax.CLim = clims;
                
                if obj.ShowStarsCheck.Value && ~isempty(obj.Sources)
                    sx = obj.Sources(:,1); sy = obj.Sources(:,2);
                    if isempty(obj.hRealTimeSrc) || ~isvalid(obj.hRealTimeSrc)
                         hold(ax, 'on');
                         obj.hRealTimeSrc = plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
                         hold(ax, 'off');
                    else
                         set(obj.hRealTimeSrc, 'XData', sx, 'YData', sy);
                    end
                elseif ~isempty(obj.hRealTimeSrc) && isvalid(obj.hRealTimeSrc)
                    set(obj.hRealTimeSrc, 'XData', [], 'YData', []);
                end
            end
            
            % === 绘制 AI 预测结果 (新增) ===
            % 使用洋红色菱形标示
            if ~isempty(obj.DLDetectedSources) && strcmp(obj.DLSwitch.Value, 'On')
                ai_x = obj.DLDetectedSources(:, 1);
                ai_y = obj.DLDetectedSources(:, 2);
                
                if isempty(obj.hDLMarkers) || ~isvalid(obj.hDLMarkers)
                    hold(ax, 'on');
                    obj.hDLMarkers = plot(ax, ai_x, ai_y, 'dm', ...
                        'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none'); 
                    hold(ax, 'off');
                else
                    set(obj.hDLMarkers, 'XData', ai_x, 'YData', ai_y, 'Visible', 'on');
                end
            else
                if ~isempty(obj.hDLMarkers) && isvalid(obj.hDLMarkers)
                    set(obj.hDLMarkers, 'Visible', 'off');
                end
            end
        end
        
        function onModeChanged(obj, src, ~)
            mode = src.Value;
            if strcmp(mode, '固定深度')
                obj.EditZFixed.Enable = 'on';
                obj.EditZStart.Enable = 'off'; obj.EditZEnd.Enable = 'off'; obj.EditZStep.Enable = 'off';
            else
                obj.EditZFixed.Enable = 'off';
                obj.EditZStart.Enable = 'on'; obj.EditZEnd.Enable = 'on'; obj.EditZStep.Enable = 'on';
            end
        end

        function onPresetParams(obj, ~, ~)
            if isempty(obj.Sources)
                uialert(obj.UIFigure, '当前没有声源，无法计算推荐值。', '提示');
                obj.EditZFixed.Value = -0.3; return;
            end
            z_coords = obj.Sources(:, 3);
            z_mean = mean(z_coords);
            z_min = min(z_coords); z_max = max(z_coords);
            obj.EditZFixed.Value = round(z_mean, 4);
            obj.EditZStart.Value = round(z_max + 0.05, 3);
            obj.EditZEnd.Value = round(z_min - 0.05, 3);
            if (z_max - z_min) > 0.01, obj.ModeSwitch.Value = '范围(MIP)';
            else, obj.ModeSwitch.Value = '固定深度'; end
            obj.onModeChanged(obj.ModeSwitch, []);
        end
        
        function onAddSource(obj, ~, ~)
            newSrc = [obj.EditX.Value, obj.EditY.Value, obj.EditZ.Value];
            obj.Sources = [obj.Sources; newSrc];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D);
            if obj.IsRealTime, obj.plot3DScene(obj.Ax3D_RT); end 
            if obj.EditMusicN.Value < size(obj.Sources, 1), obj.EditMusicN.Value = size(obj.Sources, 1); end
        end
        
        function onRemoveSource(obj, ~, ~)
            if isempty(obj.SelectedSource), return; end
            obj.Sources(obj.SelectedSource, :) = [];
            obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D);
            if obj.IsRealTime
                cla(obj.Ax3D_RT); obj.h3DSources = []; obj.plot3DScene(obj.Ax3D_RT); 
            end
        end
        
        function onClearSources(obj, ~, ~)
            obj.Sources = []; obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D);
            if obj.IsRealTime
                cla(obj.Ax3D_RT); obj.h3DSources = []; obj.plot3DScene(obj.Ax3D_RT); 
            end
        end
        
        function onSourceSelect(obj, src, ~)
            idx = find(strcmp(src.Items, src.Value));
            if ~isempty(idx)
                obj.SelectedSource = idx;
                s = obj.Sources(idx, :);
                obj.EditX.Value = s(1); obj.EditY.Value = s(2); obj.EditZ.Value = s(3);
                if obj.IsRealTime, obj.plot3DScene(obj.Ax3D_RT); else, obj.plot3DScene(obj.Ax3D); end
            end
        end
        
        function updateSourceList(obj)
            n = size(obj.Sources, 1);
            items = {};
            for i = 1:n
                items{end+1} = sprintf('Source #%d: [%.3f, %.3f, %.3f]', i, obj.Sources(i,1), obj.Sources(i,2), obj.Sources(i,3));
            end
            obj.SourceListBox.Items = items;
            if ~isempty(obj.SelectedSource) && obj.SelectedSource <= n
                obj.SourceListBox.Value = items{obj.SelectedSource};
            end
        end
        
        % === 核心算法 ===
        function runBeamforming(obj, ~, ~)
            if isempty(obj.Sources)
                uialert(obj.UIFigure, '请先添加声源!', '错误'); return;
            end
            d = uiprogressdlg(obj.UIFigure, 'Title', '计算中', 'Message', '正在计算...', 'Indeterminate', 'on');
            try
                [map, max_val, GX, GY] = obj.calculateBeamformingMap(obj.EditSNR.Value);
                obj.plotResults(GX, GY, map, max_val);
                obj.StatusLabel.Text = sprintf('%s 计算完成', obj.AlgoDropDown.Value);
                obj.RightTabs.SelectedTab = obj.TabBF;
                close(d);
            catch ME
                close(d); uialert(obj.UIFigure, ME.message, '计算错误');
            end
        end
        
        function [final_map_dB, max_val, GridX, GridY] = calculateBeamformingMap(obj, snr)
            freq = obj.EditFreq.Value;
            omega = 2 * pi * freq;
            k = omega / obj.c;
            algo = obj.AlgoDropDown.Value;
            noiseType = obj.NoiseTypeDropDown.Value;
            
            n_mics = size(obj.MicCoords, 2);
            n_src = size(obj.Sources, 1);
            if n_src == 0, error('No sources'); end
            
            % CSM 计算
            CSM = zeros(n_mics, n_mics);
            for i = 1:n_src
                src_pos = obj.Sources(i,:)';
                vec = obj.MicCoords - src_pos;
                dist = sqrt(sum(vec.^2, 1));
                signal_vec = (1./dist) .* exp(-1j * k * dist);
                CSM = CSM + (signal_vec.' * conj(signal_vec)); 
            end
            
            % 加噪
            avg_signal_power = trace(CSM) / n_mics;
            if avg_signal_power > 0, noise_power = avg_signal_power / (10^(snr/10));
            else, noise_power = 1e-12; end
            
            NoiseCSM = eye(n_mics); 
            if strcmp(noiseType, '扩散场噪声 (Diffuse)') || strcmp(noiseType, '湍流噪声 (Turbulence)')
                mics = obj.MicCoords;
                sum_sq = sum(mics.^2, 1);
                D_sq = sum_sq' + sum_sq - 2 * (mics' * mics);
                D = sqrt(max(D_sq, 0));
                
                if strcmp(noiseType, '扩散场噪声 (Diffuse)')
                    KD = k * D;
                    NoiseCSM = sin(KD) ./ (KD + 1e-12);
                    NoiseCSM(1:n_mics+1:end) = 1;
                elseif strcmp(noiseType, '湍流噪声 (Turbulence)')
                    NoiseCSM = exp(-0.5 * k * D); 
                end
            end
            CSM = CSM + noise_power * NoiseCSM;
            
            % === AI 推理通信接口 ===
            if strcmp(obj.DLSwitch.Value,'On') && ~isempty(obj.TcpClient)
                try
                    tr_val = real(trace(CSM)); if tr_val < 1e-12, tr_val = 1.0; end
                    CSM_n = CSM / tr_val;
                    
                    % 1. 获取新参数
                    val_th    = single(obj.DLThreshKnob.Value);   % 阈值
                    val_rad   = single(obj.DLRadiusKnob.Value);   % 聚合半径
                    val_egy   = single(obj.DLEnergyKnob.Value);   % 能量容忍度
                    val_debug = single(obj.DLDebugCheck.Value);    % 0或1，表示是否保存调试图
                    
                    % 2. 构造数据包: [Thr; Rad; Egy; Dbg; Real...; Imag...]
                    data = [val_th; val_rad; val_egy; val_debug; ...
                        single(real(CSM_n(:))); ...
                        single(imag(CSM_n(:)))];
                    
                    if obj.TcpClient.NumBytesAvailable > 0, read(obj.TcpClient,obj.TcpClient.NumBytesAvailable); end
                    obj.ws_send(data);
                    
                    timeout=0;
                    while obj.TcpClient.NumBytesAvailable < 2 && timeout < 200
                        pause(0.005); timeout = timeout + 1;
                    end
                    peaks = obj.ws_recv();
                    obj.DLDetectedSources = peaks;
                catch ME
                    fprintf('[WS Error] %s\n', ME.message); obj.DLDetectedSources = [];
                end
            end
            % ======================
            
            % 算法后处理
            M_core = CSM;
            func_gamma = obj.EditFuncGamma.Value;
            
            if strcmp(algo, 'MUSIC') || strcmp(algo, 'Functional')
                [V, D_eig] = eig(CSM);
                [lambda, idx] = sort(diag(real(D_eig)), 'descend');
                V = V(:, idx);
                
                if strcmp(algo, 'MUSIC')
                    n_music = min(obj.EditMusicN.Value, n_mics-1);
                    Un = V(:, n_music+1:end); 
                    M_core = Un * Un';
                elseif strcmp(algo, 'Functional')
                    noise_floor = min(lambda);
                    lambda_clean = max(lambda - noise_floor, 0); 
                    lambda_clean = max(lambda_clean, 1e-12 * max(lambda_clean));
                    D_nu = diag(lambda_clean .^ (1/func_gamma));
                    M_core = V * D_nu * V';
                end
            end
            
            % 网格生成
            if obj.IsRealTime
                grid_res = 0.004; 
            else
                grid_res = 0.002; 
            end
            
            if obj.CachedRes ~= grid_res || isempty(obj.CachedGridX)
                x_vec = -0.2:grid_res:0.2;
                y_vec = -0.2:grid_res:0.2;
                [GridX, GridY] = meshgrid(x_vec, y_vec);
                if obj.IsRealTime
                    obj.CachedGridX = GridX; obj.CachedGridY = GridY; obj.CachedRes = grid_res;
                end
            else
                GridX = obj.CachedGridX; GridY = obj.CachedGridY;
            end
            [rows, cols] = size(GridX);
            
            mode = obj.ModeSwitch.Value;
            model = obj.ModelDropDown.Value;
            
            final_map = zeros(rows, cols);
            
            if strcmp(mode, '固定深度')
                z_target = obj.EditZFixed.Value;
                GridZ = ones(size(GridX)) * z_target;
                final_map = obj.computeLayer(M_core, GridX, GridY, GridZ, k, model, algo, func_gamma);
            else 
                z_start = obj.EditZStart.Value; z_end = obj.EditZEnd.Value; z_step = obj.EditZStep.Value;
                if z_start > z_end, temp=z_start; z_start=z_end; z_end=temp; end
                z_layers = z_start:z_step:z_end;
                if isempty(z_layers), z_layers = z_start; end
                
                for idx = 1:length(z_layers)
                    z_curr = z_layers(idx);
                    GridZ = ones(size(GridX)) * z_curr;
                    layer_map = obj.computeLayer(M_core, GridX, GridY, GridZ, k, model, algo, func_gamma);
                    final_map = max(final_map, layer_map);
                end
            end
            
            final_map_dB = 10 * log10(abs(final_map) + 1e-12);
            max_val = max(final_map_dB(:));
        end
        
        function bf_map = computeLayer(obj, M_core, GX, GY, GZ, k, model, algo, gamma)
            use_gpu = obj.HasGPU;
            if use_gpu
                try
                    M_core_gpu = gpuArray(M_core);
                    GX_gpu = gpuArray(GX); GY_gpu = gpuArray(GY); GZ_gpu = gpuArray(GZ);
                    mics_gpu = gpuArray(obj.MicCoords);
                catch
                    use_gpu = false;
                end
            end
            if ~use_gpu
                 M_core_gpu = M_core;
                 GX_gpu = GX; GY_gpu = GY; GZ_gpu = GZ;
                 mics_gpu = obj.MicCoords;
            end

            [rows, cols] = size(GX_gpu);
            grid_pts = [GX_gpu(:)'; GY_gpu(:)'; GZ_gpu(:)'];
            n_mics = size(mics_gpu, 2);
            
            Rx = grid_pts(1,:) - mics_gpu(1,:)'; 
            Ry = grid_pts(2,:) - mics_gpu(2,:)'; 
            Rz = grid_pts(3,:) - mics_gpu(3,:)';
            R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); 
            
            if strcmp(model, 'near')
                W = (1./R) .* exp(-1j * k * R);
                W_norm = sqrt(sum(abs(W).^2, 1));
                W = W ./ (W_norm + 1e-12);
            else
                grid_norm = sqrt(sum(grid_pts.^2, 1));
                U = grid_pts ./ (grid_norm + 1e-12); 
                phase = k * (mics_gpu' * U);
                W = exp(-1j * phase);
                W = W / n_mics; 
            end
            
            MW = M_core_gpu * W;           
            P_vec = real(sum(conj(W) .* MW, 1)); 
            
            if strcmp(algo, 'MUSIC')
                P_vec = 1 ./ (P_vec + 1e-15);
            elseif strcmp(algo, 'Functional')
                P_vec = P_vec .^ gamma;
            end
            
            bf_map = reshape(P_vec, rows, cols);
            if use_gpu, bf_map = gather(bf_map); end
        end
        
        function plotResults(obj, GridX, GridY, map_dB, max_dB)
            ax = obj.AxBF_Map;
            cla(ax); hold(ax, 'on');
            
            algo = obj.AlgoDropDown.Value;
            if strcmp(algo, 'MUSIC'), dr = 40; 
            elseif strcmp(algo, 'Functional'), dr = 30; 
            else, dr = 15; 
            end
            
            clims = [max_dB - dr, max_dB];
            imagesc(ax, GridX(1,:), GridY(:,1), map_dB, clims);
            colormap(ax, 'jet');
            cb = colorbar(ax); 
            cb.Label.String = sprintf('SPL [dB] (%s)', algo);
            axis(ax, 'xy');
            
            % 绘制真实声源
            if obj.ShowStarsCheck.Value && ~isempty(obj.Sources)
                sx = obj.Sources(:,1); sy = obj.Sources(:,2);
                plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
            end

            % --- 新增：绘制 AI 预测结果 ---
            if ~isempty(obj.DLDetectedSources) && strcmp(obj.DLSwitch.Value, 'On')
                plot(ax, obj.DLDetectedSources(:,1), obj.DLDetectedSources(:,2), ...
                    'dm', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'none', ...
                    'DisplayName', 'AI Predicted');
            end
            % -----------------------------
            
            hold(ax, 'off');
            
            % 阵列图更新
            ax2 = obj.AxBF_Array;
            cla(ax2);
            plot(ax2, obj.MicCoords(1,:), obj.MicCoords(2,:), 'o');
            grid(ax2, 'on'); axis(ax2, 'equal'); title(ax2, '阵列 XY 投影');
        end
    end
end