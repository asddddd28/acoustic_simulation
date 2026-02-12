classdef AcousticSimRealFluentApp < handle
    % ACOUSTICSIMREALAPP 声场生成和波束成形实时仿真系统 (MATLAB版) - GPU加速版
    % 修复了 DataTips 交互报错，增加了实时布局切换功能
    % 针对 Functional Beamforming 增加了特征值降噪处理
    % 新增: 全向量化计算与 GPU 加速支持
    % 新增: 多种噪声模型支持 (传感器白噪声、扩散场噪声、湍流噪声)
    
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
        RTContainerGrid matlab.ui.container.GridLayout % 实时Tab的主容器
        RTPlotGrid      matlab.ui.container.GridLayout % 包裹两个绘图轴的网格
        Ax3D_RT         matlab.graphics.axis.Axes      % 实时模式下的 3D 轴
        AxBF_RT         matlab.graphics.axis.Axes      % 实时模式下的热力图轴
        LayoutSwitch    matlab.ui.control.Switch       % 布局切换开关
        
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
        NoiseTypeDropDown matlab.ui.control.DropDown % 新增噪声类型选择
        EditMusicN      matlab.ui.control.NumericEditField
        EditFuncGamma   matlab.ui.control.NumericEditField
        
        % 仿真数据
        MicCoords       double % 3xN 矩阵
        Sources         double % Nx3 矩阵 (x,y,z)
        SelectedSource  double = [] % 当前选中声源的索引
        
        % 绘图句柄缓存 (用于性能优化和修复DataTips bug)
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
        HasGPU          logical = false % 是否检测到 GPU
        
        % 常量
        c               double = 343; % 声速
        SampleRate      double = 51200;
        
        % 缓存网格 (避免重复计算)
        CachedGridX
        CachedGridY
        CachedRes       double = 0 % 缓存网格时的分辨率
    end
    
    methods
        function obj = AcousticSimRealFluentApp()
            % 构造函数：初始化界面和数据
            obj.checkGPU();
            obj.initData();
            obj.createUI();
            obj.initTimer();
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D); % 初始绘制
        end
        
        function delete(obj)
            % 析构函数：清理定时器
            if ~isempty(obj.SimTimer) && isvalid(obj.SimTimer)
                stop(obj.SimTimer);
                delete(obj.SimTimer);
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
            % 尝试加载外部 XML 配置
            xmlFile = 'fibonacci_32_user.xml';
            if exist(xmlFile, 'file') == 2
                obj.loadMicFromXML(xmlFile);
            else
                % fprintf('未找到麦克风配置文件 %s，使用默认螺旋阵列。\n', xmlFile);
                obj.loadDefaultArray();
            end
            
            % 初始化声源容器
            obj.Sources = [];
        end
        
        function initTimer(obj)
            % 初始化实时仿真定时器
            obj.SimTimer = timer('ExecutionMode', 'fixedRate', ...
                'Period', 0.05, ... % 提速: 20 FPS (原本10 FPS)
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
            % 初始化默认螺旋阵列
            n_mics = 32;
            r = linspace(0.01, 0.2, n_mics);
            phi = linspace(0, 2*pi*3, n_mics);
            x = r .* cos(phi);
            y = r .* sin(phi);
            z = zeros(1, n_mics);
            obj.MicCoords = [x; y; z]; 
        end
        
        function createUI(obj)
            % 创建 UI 界面
            titleStr = '声场生成与波束成形仿真系统 (GPU向量化加速版)';
            if obj.HasGPU, titleStr = [titleStr ' - GPU ON']; end
            
            obj.UIFigure = uifigure('Name', titleStr, ...
                'Position', [100, 100, 1300, 850]); 
            
            % 注册键盘事件
            obj.UIFigure.WindowKeyPressFcn = @obj.onKeyPress;
            
            obj.GridLayout = uigridlayout(obj.UIFigure, [1, 2]);
            obj.GridLayout.ColumnWidth = {360, '1x'}; 
            
            % === 左侧控制面板 ===
            obj.LeftPanel = uipanel(obj.GridLayout, 'Title', '控制面板');
            obj.LeftPanel.Layout.Row = 1;
            obj.LeftPanel.Layout.Column = 1;
            
            % 使用垂直布局管理左侧
            vbox = uigridlayout(obj.LeftPanel, [8, 1]); 
            vbox.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', '1x', 'fit'};
            vbox.Scrollable = 'on';
            
            % 1. 声源管理
            pnlSrc = uipanel(vbox, 'Title', '1. 声源管理 (WASD+QE移动)');
            pnlSrc.Layout.Row = 1;
            srcGrid = uigridlayout(pnlSrc, [2, 4]);
            srcGrid.RowHeight = {'fit', 'fit'};
            
            uilabel(srcGrid, 'Text', 'X:');
            obj.EditX = uieditfield(srcGrid, 'numeric', 'Value', 0.0);
            uilabel(srcGrid, 'Text', 'Y:');
            obj.EditY = uieditfield(srcGrid, 'numeric', 'Value', 0.0);
            uilabel(srcGrid, 'Text', 'Z:');
            obj.EditZ = uieditfield(srcGrid, 'numeric', 'Value', -0.3);
            
            uibutton(srcGrid, 'Text', '添加', 'ButtonPushedFcn', @obj.onAddSource);
            uibutton(srcGrid, 'Text', '删除选中', 'ButtonPushedFcn', @obj.onRemoveSource);
            uibutton(srcGrid, 'Text', '清空', 'ButtonPushedFcn', @obj.onClearSources);
            
            % 列表框
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
            obj.ModeSwitch = uiswitch(scanGrid, 'Items', {'固定深度', '范围(MIP)'}, ...
                'ValueChangedFcn', @obj.onModeChanged);
            
            lblFix = uilabel(scanGrid, 'Text', '固定深度 Z(m):');
            obj.EditZFixed = uieditfield(scanGrid, 'numeric', 'Value', -0.3);
            
            lblRange = uilabel(scanGrid, 'Text', '范围 Z(start/end/step):');
            rangeBox = uigridlayout(scanGrid, [1, 3]);
            rangeBox.Padding = [0 0 0 0];
            rangeBox.Layout.Column = 2;
            obj.EditZStart = uieditfield(rangeBox, 'numeric', 'Value', -0.1, 'Enable', 'off');
            obj.EditZEnd = uieditfield(rangeBox, 'numeric', 'Value', -0.5, 'Enable', 'off');
            obj.EditZStep = uieditfield(rangeBox, 'numeric', 'Value', 0.02, 'Enable', 'off');
            
            presetBtn = uibutton(scanGrid, 'Text', '从当前声源预设参数', ...
                'ButtonPushedFcn', @obj.onPresetParams);
            presetBtn.Layout.Column = [1 2];
            
            % 3. 算法参数
            pnlAlgo = uipanel(vbox, 'Title', '3. 算法运行');
            pnlAlgo.Layout.Row = 4;
            algoGrid = uigridlayout(pnlAlgo, [7, 2]); % 增加一行
            algoGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit'};
            
            uilabel(algoGrid, 'Text', '中心频率 (Hz):');
            obj.EditFreq = uieditfield(algoGrid, 'numeric', 'Value', 8000);
            
            uilabel(algoGrid, 'Text', '信噪比 SNR (dB):');
            obj.EditSNR = uieditfield(algoGrid, 'numeric', 'Value', 30, 'Limits', [-20 100]);
            
            % 新增：噪声类型选择
            uilabel(algoGrid, 'Text', '噪声模型:');
            obj.NoiseTypeDropDown = uidropdown(algoGrid, 'Items', ...
                {'传感器白噪声 (White)', '扩散场噪声 (Diffuse)', '湍流噪声 (Turbulence)'}, ...
                'Value', '传感器白噪声 (White)');
            
            uilabel(algoGrid, 'Text', '算法类型:');
            obj.AlgoDropDown = uidropdown(algoGrid, 'Items', {'Conventional', 'MUSIC', 'Functional'}, ...
                'Value', 'Conventional');
            
            paramGrid = uigridlayout(algoGrid, [1, 4]);
            paramGrid.Layout.Column = [1 2];
            paramGrid.Padding = [0 0 0 0];
            uilabel(paramGrid, 'Text', 'MUSIC源数:');
            obj.EditMusicN = uieditfield(paramGrid, 'numeric', 'Value', 1, 'Limits', [1 100], 'RoundFractionalValues', 'on');
            uilabel(paramGrid, 'Text', 'Gamma:');
            obj.EditFuncGamma = uieditfield(paramGrid, 'numeric', 'Value', 10, 'Limits', [1 1000]);
            
            runBtn = uibutton(algoGrid, 'Text', '单次运行', ...
                'BackgroundColor', [0.3, 0.6, 1.0], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @obj.runBeamforming);
            runBtn.Layout.Column = [1 2];
            
            % 实时开关
            uilabel(algoGrid, 'Text', '实时仿真模式:', 'FontWeight', 'bold', 'FontColor', [0.8 0.2 0]);
            obj.RealTimeSwitch = uiswitch(algoGrid, 'Items', {'Off', 'On'}, ...
                'ValueChangedFcn', @obj.onRealTimeToggle);
            
            % 状态栏
            accelType = 'CPU Vector';
            if obj.HasGPU, accelType = 'GPU Accelerated'; end
            obj.StatusLabel = uilabel(vbox, 'Text', sprintf('就绪 [%s] | 提示: WASD移动声源', accelType));
            obj.StatusLabel.Layout.Row = 8;
            obj.StatusLabel.FontColor = [0.4 0.4 0.4];
            
            % === 右侧绘图区 ===
            obj.RightTabs = uitabgroup(obj.GridLayout);
            obj.RightTabs.Layout.Row = 1;
            obj.RightTabs.Layout.Column = 2;
            
            % Tab 1: 3D 视图 (常规)
            obj.Tab3D = uitab(obj.RightTabs, 'Title', '3D 设置视图');
            gl3d = uigridlayout(obj.Tab3D, [1,1]);
            obj.Ax3D = uiaxes(gl3d);
            xlabel(obj.Ax3D, 'X (m)'); ylabel(obj.Ax3D, 'Y (m)'); zlabel(obj.Ax3D, 'Z (m)');
            grid(obj.Ax3D, 'on');
            title(obj.Ax3D, '声源位置配置 (Setup View)');
            view(obj.Ax3D, obj.ViewAzim, obj.ViewElev);
            
            % Tab 2: 波束成形结果 (常规)
            obj.TabBF = uitab(obj.RightTabs, 'Title', '波束成形热力图');
            glbf = uigridlayout(obj.TabBF, [2, 1]);
            glbf.RowHeight = {'2x', '1x'};
            obj.AxBF_Map = uiaxes(glbf);
            title(obj.AxBF_Map, '波束成形结果 (Result View)');
            axis(obj.AxBF_Map, 'equal');
            obj.AxBF_Array = uiaxes(glbf);
            title(obj.AxBF_Array, '麦克风布局');
            
            % Tab 3: 实时仿真模式 (新增，包含左右分屏/上下分屏)
            obj.TabRealTime = uitab(obj.RightTabs, 'Title', '实时仿真 (Real-time)');
            
            % 实时Tab的主布局: 顶部工具栏 + 绘图区
            obj.RTContainerGrid = uigridlayout(obj.TabRealTime, [2, 1]);
            obj.RTContainerGrid.RowHeight = {'fit', '1x'};
            
            % 顶部工具栏
            toolbarGrid = uigridlayout(obj.RTContainerGrid, [1, 3]);
            toolbarGrid.ColumnWidth = {'fit', 'fit', '1x'};
            toolbarGrid.Layout.Row = 1;
            
            uilabel(toolbarGrid, 'Text', '界面布局:');
            obj.LayoutSwitch = uiswitch(toolbarGrid, 'Items', {'左右分屏', '上下分屏'}, ...
                'ValueChangedFcn', @obj.onLayoutChange);
            
            % 绘图区容器 (可以动态改变行列数)
            obj.RTPlotGrid = uigridlayout(obj.RTContainerGrid, [1, 2]); % 默认左右
            obj.RTPlotGrid.Layout.Row = 2;
            obj.RTPlotGrid.ColumnWidth = {'1x', '1x'};
            
            % 左/上：实时 3D 视图
            obj.Ax3D_RT = uiaxes(obj.RTPlotGrid);
            obj.Ax3D_RT.Layout.Row = 1;
            obj.Ax3D_RT.Layout.Column = 1;
            title(obj.Ax3D_RT, '实时空间视图 (可交互)');
            xlabel(obj.Ax3D_RT, 'X'); ylabel(obj.Ax3D_RT, 'Y'); zlabel(obj.Ax3D_RT, 'Z');
            grid(obj.Ax3D_RT, 'on');
            
            % 右/下：实时热力图
            obj.AxBF_RT = uiaxes(obj.RTPlotGrid);
            obj.AxBF_RT.Layout.Row = 1;
            obj.AxBF_RT.Layout.Column = 2;
            title(obj.AxBF_RT, '实时波束成形');
            axis(obj.AxBF_RT, 'equal');
        end
        
        % === 布局切换逻辑 ===
        function onLayoutChange(obj, src, ~)
            val = src.Value;
            if strcmp(val, '左右分屏')
                % 设置为 1行2列
                obj.RTPlotGrid.RowHeight = {'1x'};
                obj.RTPlotGrid.ColumnWidth = {'1x', '1x'};
                
                % 重新指定 Axes 位置
                obj.Ax3D_RT.Layout.Row = 1;
                obj.Ax3D_RT.Layout.Column = 1;
                obj.AxBF_RT.Layout.Row = 1;
                obj.AxBF_RT.Layout.Column = 2;
            else
                % 设置为 2行1列
                obj.RTPlotGrid.RowHeight = {'1x', '1x'};
                obj.RTPlotGrid.ColumnWidth = {'1x'};
                
                % 重新指定 Axes 位置
                obj.Ax3D_RT.Layout.Row = 1;
                obj.Ax3D_RT.Layout.Column = 1;
                obj.AxBF_RT.Layout.Row = 2;
                obj.AxBF_RT.Layout.Column = 1;
            end
        end
        
        % === 实时模式逻辑 ===
        
        function onRealTimeToggle(obj, src, ~)
            if strcmp(src.Value, 'On')
                % 开启实时模式
                obj.IsRealTime = true;
                
                % 1. 切换到实时 Tab
                obj.RightTabs.SelectedTab = obj.TabRealTime;
                
                % 2. 预先清空句柄缓存，强制重绘一次基础元素
                obj.hMapImage = [];
                obj.hRealTimeSrc = [];
                obj.hRealTimeMics = [];
                
                % 3. 同步视角并初始化 3D 绘图
                [az, el] = view(obj.Ax3D);
                view(obj.Ax3D_RT, az, el);
                obj.plot3DScene(obj.Ax3D_RT); % 初始化场景
                
                % 4. 启动定时器
                start(obj.SimTimer);
                
                modeStr = 'CPU Vector';
                if obj.HasGPU, modeStr = 'GPU Accelerated'; end
                obj.StatusLabel.Text = sprintf('实时仿真运行中 (%s)... WASD移动', modeStr);
                
            else
                % 关闭实时模式
                obj.IsRealTime = false;
                stop(obj.SimTimer);
                
                % 1. 切换回常规 Tab (比如结果页)
                obj.RightTabs.SelectedTab = obj.TabBF;
                obj.StatusLabel.Text = '实时仿真已停止';
            end
        end
        
        function onRealTimeStep(obj, ~, ~)
            % 定时器回调：执行一次快速仿真
            try
                % 添加噪声波动 (模拟 SNR 波动 +/- 2dB)
                baseSNR = obj.EditSNR.Value;
                currentSNR = baseSNR + 2 * randn(); 
                
                % 调用核心计算
                [map, max_val, GX, GY] = obj.calculateBeamformingMap(currentSNR);
                
                % 刷新实时热力图 (AxBF_RT)
                obj.plotResultsRT(GX, GY, map, max_val);
                
            catch ME
                % 出错则停止
                stop(obj.SimTimer);
                obj.RealTimeSwitch.Value = 'Off';
                obj.onRealTimeToggle(obj.RealTimeSwitch, []);
                uialert(obj.UIFigure, ME.message, '实时仿真出错');
            end
        end
        
        % === 交互回调函数 ===
        
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
            
            % 更新数据
            obj.Sources(obj.SelectedSource, :) = obj.Sources(obj.SelectedSource, :) + [dx, dy, dz];
            s = obj.Sources(obj.SelectedSource, :);
            obj.EditX.Value = s(1);
            obj.EditY.Value = s(2);
            obj.EditZ.Value = s(3);
            
            obj.updateSourceList();
            
            % 关键：根据当前模式更新对应的 3D 视图
            if obj.IsRealTime
                % 实时模式下调用优化后的绘图，不再 cla
                obj.plot3DScene(obj.Ax3D_RT);
            else
                obj.plot3DScene(obj.Ax3D);
            end
        end

        function plot3DScene(obj, ax)
            % 绘制 3D 场景
            % 优化策略：如果是在实时模式，避免使用 cla，而是更新对象属性
            
            isRT = (ax == obj.Ax3D_RT);
            
            % 只有非实时模式才强制清除，保证干净的开始
            if ~isRT
                cla(ax);
                hold(ax, 'on');
            else
                % 实时模式，确保 hold on
                hold(ax, 'on');
            end
            
            % 1. 绘制麦克风 (静态，通常只画一次)
            mx = obj.MicCoords(1,:);
            my = obj.MicCoords(2,:);
            mz = obj.MicCoords(3,:);
            
            if ~isRT
                % 常规模式：直接画
                scatter3(ax, mx, my, mz, 50, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
                % 辅助圆
                theta = 0:0.1:2*pi;
                max_r = max(sqrt(mx.^2 + my.^2)) * 1.1;
                fill3(ax, max_r*cos(theta), max_r*sin(theta), zeros(size(theta)), ...
                    [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'k');
            elseif isempty(obj.hRealTimeMics) || ~isvalid(obj.hRealTimeMics)
                % 实时模式初始化
                obj.hRealTimeMics = scatter3(ax, mx, my, mz, 50, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
                theta = 0:0.1:2*pi;
                max_r = max(sqrt(mx.^2 + my.^2)) * 1.1;
                fill3(ax, max_r*cos(theta), max_r*sin(theta), zeros(size(theta)), ...
                    [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'k');
            end
            
            % 2. 绘制声源 (动态)
            nSrc = size(obj.Sources, 1);
            
            if ~isRT
                % === 常规模式：全重画 ===
                if nSrc > 0
                    sx = obj.Sources(:,1);
                    sy = obj.Sources(:,2);
                    sz = obj.Sources(:,3);
                    % 垂线
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
                axis(ax, 'equal');
                zlim(ax, [-0.6 0.1]); 
                hold(ax, 'off');
                
            else
                % === 实时模式：更新对象属性 ===
                % 为简化实时逻辑，如果声源数量变化，我们不得不删掉 scatter 重建
                % 但如果只是位置变化，我们更新 XData/YData
                
                sx = obj.Sources(:,1);
                sy = obj.Sources(:,2);
                sz = obj.Sources(:,3);
                
                if isempty(obj.h3DSources) || ~isvalid(obj.h3DSources) || length(obj.h3DSources.XData) ~= nSrc
                    % 初始化或重建
                    delete(findobj(ax, 'Type', 'Scatter', 'Marker', 'p')); % 清除旧的声源点
                    delete(findobj(ax, 'Type', 'Line', 'LineStyle', '--')); % 清除旧垂线
                    
                    if nSrc > 0
                        colors = repmat([1 0 0], nSrc, 1);
                        sizes = repmat(100, nSrc, 1);
                        if ~isempty(obj.SelectedSource) && obj.SelectedSource <= nSrc
                            colors(obj.SelectedSource, :) = [1 0.6 0]; 
                            sizes(obj.SelectedSource) = 200;
                        end
                        obj.h3DSources = scatter3(ax, sx, sy, sz, sizes, colors, 'filled', 'p');
                    end
                else
                    % 更新现有句柄
                    if nSrc > 0
                        set(obj.h3DSources, 'XData', sx, 'YData', sy, 'ZData', sz);
                        % 更新高亮
                        colors = repmat([1 0 0], nSrc, 1);
                        sizes = repmat(100, nSrc, 1);
                        if ~isempty(obj.SelectedSource) && obj.SelectedSource <= nSrc
                            colors(obj.SelectedSource, :) = [1 0.6 0]; 
                            sizes(obj.SelectedSource) = 200;
                        end
                        set(obj.h3DSources, 'CData', colors, 'SizeData', sizes);
                    end
                end
                
                axis(ax, 'equal');
                zlim(ax, [-0.6 0.1]); 
            end
        end
        
        function plotResultsRT(obj, GridX, GridY, map_dB, max_dB)
            % 专用于实时模式的轻量化绘图 (修复 DataTips 问题的关键)
            ax = obj.AxBF_RT;
            
            algo = obj.AlgoDropDown.Value;
            if strcmp(algo, 'MUSIC'), dr = 40; 
            elseif strcmp(algo, 'Functional'), dr = 30; 
            else, dr = 15; end
            
            clims = [max_dB - dr, max_dB];
            
            % === 修复核心：如果图像对象已存在，直接更新 CData，不使用 cla ===
            if isempty(obj.hMapImage) || ~isvalid(obj.hMapImage)
                % 首次绘制
                cla(ax); % 仅在第一次清理
                hold(ax, 'on');
                obj.hMapImage = imagesc(ax, GridX(1,:), GridY(:,1), map_dB);
                colormap(ax, 'jet');
                axis(ax, 'xy');
                axis(ax, 'equal');
                obj.hMapImage.CDataMapping = 'scaled';
                ax.CLim = clims;
                
                % 绘制真实位置辅助点 (保存句柄)
                if obj.ShowStarsCheck.Value && ~isempty(obj.Sources)
                   sx = obj.Sources(:,1);
                   sy = obj.Sources(:,2);
                   obj.hRealTimeSrc = plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
                end
                hold(ax, 'off');
            else
                % 更新数据
                set(obj.hMapImage, 'CData', map_dB);
                ax.CLim = clims; % 更新色标范围
                
                % 更新辅助点
                if obj.ShowStarsCheck.Value && ~isempty(obj.Sources)
                    sx = obj.Sources(:,1);
                    sy = obj.Sources(:,2);
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
        end
        
        % ... (其他保持不变的回调函数) ...
        
        function onModeChanged(obj, src, ~)
            mode = src.Value;
            if strcmp(mode, '固定深度')
                obj.EditZFixed.Enable = 'on';
                obj.EditZStart.Enable = 'off';
                obj.EditZEnd.Enable = 'off';
                obj.EditZStep.Enable = 'off';
            else
                obj.EditZFixed.Enable = 'off';
                obj.EditZStart.Enable = 'on';
                obj.EditZEnd.Enable = 'on';
                obj.EditZStep.Enable = 'on';
            end
        end

        function onPresetParams(obj, ~, ~)
            if isempty(obj.Sources)
                uialert(obj.UIFigure, '当前没有声源，无法计算推荐值。', '提示');
                obj.EditZFixed.Value = -0.3;
                return;
            end
            z_coords = obj.Sources(:, 3);
            z_mean = mean(z_coords);
            z_min = min(z_coords);
            z_max = max(z_coords);
            obj.EditZFixed.Value = round(z_mean, 4);
            obj.EditZStart.Value = round(z_max + 0.05, 3);
            obj.EditZEnd.Value = round(z_min - 0.05, 3);
            if (z_max - z_min) > 0.01
                obj.ModeSwitch.Value = '范围(MIP)';
            else
                obj.ModeSwitch.Value = '固定深度';
            end
            obj.onModeChanged(obj.ModeSwitch, []);
        end
        
        function onAddSource(obj, ~, ~)
            newSrc = [obj.EditX.Value, obj.EditY.Value, obj.EditZ.Value];
            obj.Sources = [obj.Sources; newSrc];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D); % 添加时更新主视图
            if obj.IsRealTime, obj.plot3DScene(obj.Ax3D_RT); end 
            if obj.EditMusicN.Value < size(obj.Sources, 1)
                obj.EditMusicN.Value = size(obj.Sources, 1);
            end
        end
        
        function onRemoveSource(obj, ~, ~)
            idx = obj.SourceListBox.Value;
            if isempty(idx) || isempty(obj.SelectedSource), return; end
            obj.Sources(obj.SelectedSource, :) = [];
            obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D);
            if obj.IsRealTime
                % 强制重建 3D 视图，因为对象数量变了
                cla(obj.Ax3D_RT); 
                obj.h3DSources = []; % 重置句柄
                obj.plot3DScene(obj.Ax3D_RT); 
            end
        end
        
        function onClearSources(obj, ~, ~)
            obj.Sources = [];
            obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene(obj.Ax3D);
            if obj.IsRealTime
                cla(obj.Ax3D_RT);
                obj.h3DSources = [];
                obj.plot3DScene(obj.Ax3D_RT); 
            end
        end
        
        function onSourceSelect(obj, src, ~)
            idx = find(strcmp(src.Items, src.Value));
            if ~isempty(idx)
                obj.SelectedSource = idx;
                s = obj.Sources(idx, :);
                obj.EditX.Value = s(1);
                obj.EditY.Value = s(2);
                obj.EditZ.Value = s(3);
                % 选择时更新对应的视图
                if obj.IsRealTime
                    obj.plot3DScene(obj.Ax3D_RT);
                else
                    obj.plot3DScene(obj.Ax3D); 
                end
            end
        end
        
        function updateSourceList(obj)
            n = size(obj.Sources, 1);
            items = {};
            for i = 1:n
                items{end+1} = sprintf('Source #%d: [%.3f, %.3f, %.3f]', ...
                    i, obj.Sources(i,1), obj.Sources(i,2), obj.Sources(i,3));
            end
            obj.SourceListBox.Items = items;
            if ~isempty(obj.SelectedSource) && obj.SelectedSource <= n
                obj.SourceListBox.Value = items{obj.SelectedSource};
            end
        end
        
        % === 核心算法：波束成形 ===
        
        function runBeamforming(obj, ~, ~)
            % 单次运行按钮回调
            if isempty(obj.Sources)
                uialert(obj.UIFigure, '请先添加声源!', '错误');
                return;
            end
            
            d = uiprogressdlg(obj.UIFigure, 'Title', '计算中', 'Message', '正在计算...', 'Indeterminate', 'on');
            try
                % 调用计算核心
                [map, max_val, GX, GY] = obj.calculateBeamformingMap(obj.EditSNR.Value);
                obj.plotResults(GX, GY, map, max_val);
                
                obj.StatusLabel.Text = sprintf('%s 计算完成', obj.AlgoDropDown.Value);
                obj.RightTabs.SelectedTab = obj.TabBF;
                close(d);
            catch ME
                close(d);
                uialert(obj.UIFigure, ME.message, '计算错误');
            end
        end
        
        function [final_map_dB, max_val, GridX, GridY] = calculateBeamformingMap(obj, snr)
            % 核心计算函数
            
            % 1. 参数准备
            freq = obj.EditFreq.Value;
            omega = 2 * pi * freq;
            k = omega / obj.c;
            algo = obj.AlgoDropDown.Value;
            noiseType = obj.NoiseTypeDropDown.Value;
            
            n_mics = size(obj.MicCoords, 2);
            n_src = size(obj.Sources, 1);
            
            if n_src == 0
                error('No sources');
            end
            
            % 2. 计算互谱矩阵 (CSM)
            CSM = zeros(n_mics, n_mics);
            for i = 1:n_src
                src_pos = obj.Sources(i,:)';
                vec = obj.MicCoords - src_pos;
                dist = sqrt(sum(vec.^2, 1));
                signal_vec = (1./dist) .* exp(-1j * k * dist);
                CSM = CSM + (signal_vec.' * conj(signal_vec)); 
            end
            
            % 3. 添加噪声 (支持多种模型)
            avg_signal_power = trace(CSM) / n_mics;
            if avg_signal_power > 0
                noise_power = avg_signal_power / (10^(snr/10));
            else
                noise_power = 1e-12; 
            end
            
            % 生成噪声矩阵 NoiseCSM (归一化功率)
            NoiseCSM = eye(n_mics); % 默认白噪声
            
            if strcmp(noiseType, '扩散场噪声 (Diffuse)') || strcmp(noiseType, '湍流噪声 (Turbulence)')
                % 计算麦克风间距矩阵 (这里简单起见使用 CPU 计算，因为 n_mics 很小)
                mics = obj.MicCoords;
                % 向量化计算距离矩阵 D (Nm x Nm)
                % D_sq = sum(x.^2) + sum(y.^2)' - 2*x*y'
                sum_sq = sum(mics.^2, 1);
                D_sq = sum_sq' + sum_sq - 2 * (mics' * mics);
                D = sqrt(max(D_sq, 0));
                
                if strcmp(noiseType, '扩散场噪声 (Diffuse)')
                    % sinc(kd) = sin(kd)/kd
                    KD = k * D;
                    % 处理 KD=0 的情况 (对角线)
                    NoiseCSM = sin(KD) ./ (KD + 1e-12);
                    NoiseCSM(1:n_mics+1:end) = 1; % 确保对角线为1
                elseif strcmp(noiseType, '湍流噪声 (Turbulence)')
                    % 指数衰减相关性 exp(-alpha * k * d)
                    % alpha 系数模拟相关长度，这里取 0.5 作为示例
                    NoiseCSM = exp(-0.5 * k * D); 
                end
            end
            
            CSM = CSM + noise_power * NoiseCSM;
            
            % 4. 算法预处理
            M_core = CSM;
            func_gamma = obj.EditFuncGamma.Value;
            
            if strcmp(algo, 'MUSIC') || strcmp(algo, 'Functional')
                [V, D] = eig(CSM);
                [lambda, idx] = sort(diag(real(D)), 'descend');
                V = V(:, idx);
                
                if strcmp(algo, 'MUSIC')
                    n_music = min(obj.EditMusicN.Value, n_mics-1);
                    Un = V(:, n_music+1:end); 
                    M_core = Un * Un';
                elseif strcmp(algo, 'Functional')
                    % FIX: 减去噪声底 (最小特征值)，消除全方向背景噪声的影响
                    noise_floor = min(lambda);
                    lambda_clean = max(lambda - noise_floor, 0); 
                    lambda_clean = max(lambda_clean, 1e-12 * max(lambda_clean));
                    D_nu = diag(lambda_clean .^ (1/func_gamma));
                    M_core = V * D_nu * V';
                end
            end
            
            % 5. 扫描配置 (大幅提升分辨率)
            if obj.IsRealTime
                grid_res = 0.004; % 实时: 4mm (约 1万点) - 得益于 GPU/Vectorization
            else
                grid_res = 0.002; % 单次: 2mm (约 4万点)
            end
            
            % 缓存网格优化
            if obj.CachedRes ~= grid_res || isempty(obj.CachedGridX) || size(obj.CachedGridX, 1) ~= length(-0.2:grid_res:0.2)
                x_vec = -0.2:grid_res:0.2;
                y_vec = -0.2:grid_res:0.2;
                [GridX, GridY] = meshgrid(x_vec, y_vec);
                if obj.IsRealTime
                    obj.CachedGridX = GridX;
                    obj.CachedGridY = GridY;
                    obj.CachedRes = grid_res;
                end
            else
                GridX = obj.CachedGridX;
                GridY = obj.CachedGridY;
            end
            [rows, cols] = size(GridX);
            
            % 6. 计算层
            mode = obj.ModeSwitch.Value;
            model = obj.ModelDropDown.Value;
            
            final_map = zeros(rows, cols);
            
            if strcmp(mode, '固定深度')
                z_target = obj.EditZFixed.Value;
                GridZ = ones(size(GridX)) * z_target;
                final_map = obj.computeLayer(M_core, GridX, GridY, GridZ, k, model, algo, func_gamma);
            else 
                % 范围扫描 (MIP)
                z_start = obj.EditZStart.Value;
                z_end = obj.EditZEnd.Value;
                z_step = obj.EditZStep.Value;
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
            % 全向量化 & GPU 加速计算层
            
            use_gpu = obj.HasGPU;
            
            % 数据准备: 移至 GPU (如果可用)
            if use_gpu
                try
                    M_core_gpu = gpuArray(M_core);
                    GX_gpu = gpuArray(GX);
                    GY_gpu = gpuArray(GY);
                    GZ_gpu = gpuArray(GZ);
                    mics_gpu = gpuArray(obj.MicCoords);
                catch
                    use_gpu = false;
                end
            end
            
            % 回退到 CPU 数据
            if ~use_gpu
                 M_core_gpu = M_core;
                 GX_gpu = GX;
                 GY_gpu = GY;
                 GZ_gpu = GZ;
                 mics_gpu = obj.MicCoords;
            end

            [rows, cols] = size(GX_gpu);
            
            % 展平网格点 (3 x Np)
            grid_pts = [GX_gpu(:)'; GY_gpu(:)'; GZ_gpu(:)'];
            n_mics = size(mics_gpu, 2);
            
            % 1. 计算距离矩阵 R (Nm x Np)
            % 使用显式扩展 (Broadcasting) 计算所有麦克风到所有网格点的距离
            % mics_gpu(1,:) 是 1 x Nm -> 转置为 Nm x 1
            % grid_pts(1,:) 是 1 x Np
            Rx = grid_pts(1,:) - mics_gpu(1,:)'; 
            Ry = grid_pts(2,:) - mics_gpu(2,:)'; 
            Rz = grid_pts(3,:) - mics_gpu(3,:)';
            
            R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % 结果: Nm x Np
            
            % 2. 计算导向矢量 W (Nm x Np)
            if strcmp(model, 'near')
                % 球面波模型: (1/r) * exp(-jkr)
                W = (1./R) .* exp(-1j * k * R);
                % 列归一化 (Match Acoular logic)
                W_norm = sqrt(sum(abs(W).^2, 1));
                W = W ./ (W_norm + 1e-12);
            else
                % 平面波模型 (Far field)
                grid_norm = sqrt(sum(grid_pts.^2, 1));
                U = grid_pts ./ (grid_norm + 1e-12); % 3 x Np
                
                % phase = k * (mics' * U) -> (Nm x 3) * (3 x Np) -> Nm x Np
                phase = k * (mics_gpu' * U);
                W = exp(-1j * phase);
                W = W / n_mics; 
            end
            
            % 3. 波束成形功率计算 P = w' * C * w
            % 向量化技巧: diag(W' * M * W) 等价于 sum(conj(W) .* (M * W), 1)
            % 这避免了计算巨大的 (Np x Np) 矩阵，只计算对角线元素
            
            MW = M_core_gpu * W;           % (Nm x Nm) * (Nm x Np) -> Nm x Np
            P_vec = real(sum(conj(W) .* MW, 1)); % Element-wise multiply -> Sum cols -> 1 x Np
            
            % 4. 算法后处理
            if strcmp(algo, 'MUSIC')
                P_vec = 1 ./ (P_vec + 1e-15);
            elseif strcmp(algo, 'Functional')
                P_vec = P_vec .^ gamma;
            end
            
            % 5. 重塑并取回数据
            bf_map = reshape(P_vec, rows, cols);
            if use_gpu
                bf_map = gather(bf_map);
            end
        end
        
        function plotResults(obj, GridX, GridY, map_dB, max_dB)
            ax = obj.AxBF_Map;
            cla(ax);
            hold(ax, 'on');
            
            algo = obj.AlgoDropDown.Value;
            if strcmp(algo, 'MUSIC')
                dr = 40; 
            elseif strcmp(algo, 'Functional')
                dr = 30; 
            else
                dr = 15; 
            end
            
            clims = [max_dB - dr, max_dB];
            imagesc(ax, GridX(1,:), GridY(:,1), map_dB, clims);
            colormap(ax, 'jet');
            
            c = colorbar(ax);
            c.Label.String = sprintf('SPL [dB] (%s)', algo);
            
            axis(ax, 'xy');
            
            if obj.ShowStarsCheck.Value
                sx = obj.Sources(:,1);
                sy = obj.Sources(:,2);
                plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
            end
            hold(ax, 'off');
            
            ax2 = obj.AxBF_Array;
            cla(ax2);
            plot(ax2, obj.MicCoords(1,:), obj.MicCoords(2,:), 'o');
            grid(ax2, 'on');
            axis(ax2, 'equal');
            title(ax2, '阵列 XY 投影');
        end
    end
end