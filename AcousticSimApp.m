classdef AcousticSimApp < handle
    % ACOUSTICSIMAPP 声场生成和波束成形实时仿真系统 (MATLAB版)
    % 移植自 Python Acoular 仿真程序
    % 功能：
    % 1. 快捷键: WASD (平面移动) + QE (垂直移动) + Delete (删除)
    % 2. 3D 视图与热力图显示
    % 3. 支持近场/远场模型切换
    % 4. 支持固定深度/范围扫描(MIP)
    % 5. 新增 MUSIC 和 Functional Beamforming 高级算法
    % 6. 支持 SNR 信噪比调节
    
    properties
        % UI 组件
        UIFigure        matlab.ui.Figure
        GridLayout      matlab.ui.container.GridLayout
        LeftPanel       matlab.ui.container.Panel
        RightTabs       matlab.ui.container.TabGroup
        Tab3D           matlab.ui.container.Tab
        TabBF           matlab.ui.container.Tab
        Ax3D            matlab.graphics.axis.Axes
        AxBF_Map        matlab.graphics.axis.Axes
        AxBF_Array      matlab.graphics.axis.Axes
        SourceListBox   matlab.ui.control.ListBox
        StatusLabel     matlab.ui.control.Label
        
        % 输入控件
        EditX, EditY, EditZ
        EditFreq, EditZFixed, EditZStart, EditZEnd, EditZStep
        EditSNR         matlab.ui.control.NumericEditField % 新增 SNR 控件
        ModeSwitch      matlab.ui.control.Switch
        ModelDropDown   matlab.ui.control.DropDown
        ShowStarsCheck  matlab.ui.control.CheckBox
        
        % 算法控件
        AlgoDropDown    matlab.ui.control.DropDown
        EditMusicN      matlab.ui.control.NumericEditField
        EditFuncGamma   matlab.ui.control.NumericEditField
        
        % 仿真数据
        MicCoords       double % 3xN 矩阵
        Sources         double % Nx3 矩阵 (x,y,z)
        SelectedSource  double = [] % 当前选中声源的索引
        
        % 视图状态
        ViewAzim        double = 45
        ViewElev        double = 20
        
        % 常量
        c               double = 343; % 声速
        SampleRate      double = 51200;
    end
    
    methods
        function obj = AcousticSimApp()
            % 构造函数：初始化界面和数据
            obj.initData();
            obj.createUI();
            obj.updateSourceList();
            obj.plot3DScene();
        end
        
        function initData(obj)
            % 尝试加载外部 XML 配置
            xmlFile = 'fibonacci_32_user.xml';
            if exist(xmlFile, 'file') == 2
                obj.loadMicFromXML(xmlFile);
            else
                fprintf('未找到麦克风配置文件 %s，使用默认螺旋阵列。\n', xmlFile);
                obj.loadDefaultArray();
            end
            
            % 初始化声源容器
            obj.Sources = [];
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
                    % getAttribute 返回的是 Java String 或 char 数组，确保转换
                    x(k+1) = str2double(char(item.getAttribute('x')));
                    y(k+1) = str2double(char(item.getAttribute('y')));
                    z(k+1) = str2double(char(item.getAttribute('z')));
                end
                
                obj.MicCoords = [x; y; z];
                fprintf('成功加载麦克风阵列: %s (%d 个麦克风)\n', filename, n_mics);
            catch ME
                fprintf('解析 XML 失败: %s\n', ME.message);
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
            obj.MicCoords = [x; y; z]; % 3xN
        end
        
        function createUI(obj)
            % 创建 UI 界面
            obj.UIFigure = uifigure('Name', '声场生成与波束成形仿真系统 (MATLAB版)', ...
                'Position', [100, 100, 1200, 850]);
            
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
            vbox.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', '1x'};
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
            obj.SourceListBox.Layout.Column = 1;
            
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
            
            % 固定深度参数
            lblFix = uilabel(scanGrid, 'Text', '固定深度 Z(m):');
            obj.EditZFixed = uieditfield(scanGrid, 'numeric', 'Value', -0.3);
            
            % 范围参数 (初始隐藏或禁用，这里简单起见全部显示，逻辑控制状态)
            lblRange = uilabel(scanGrid, 'Text', '范围 Z(start/end/step):');
            rangeBox = uigridlayout(scanGrid, [1, 3]);
            rangeBox.Padding = [0 0 0 0];
            rangeBox.Layout.Column = 2;
            obj.EditZStart = uieditfield(rangeBox, 'numeric', 'Value', -0.1, 'Enable', 'off');
            obj.EditZEnd = uieditfield(rangeBox, 'numeric', 'Value', -0.5, 'Enable', 'off');
            obj.EditZStep = uieditfield(rangeBox, 'numeric', 'Value', 0.02, 'Enable', 'off');
            
            % 预设按钮
            presetBtn = uibutton(scanGrid, 'Text', '从当前声源预设参数', ...
                'ButtonPushedFcn', @obj.onPresetParams);
            presetBtn.Layout.Column = [1 2];
            
            % 3. 算法参数
            pnlAlgo = uipanel(vbox, 'Title', '3. 算法运行');
            pnlAlgo.Layout.Row = 4;
            algoGrid = uigridlayout(pnlAlgo, [5, 2]); % 增加一行用于 SNR
            algoGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit'};
            
            uilabel(algoGrid, 'Text', '中心频率 (Hz):');
            obj.EditFreq = uieditfield(algoGrid, 'numeric', 'Value', 8000);
            
            % 新增 SNR 设置
            uilabel(algoGrid, 'Text', '信噪比 SNR (dB):');
            obj.EditSNR = uieditfield(algoGrid, 'numeric', 'Value', 30, 'Limits', [-20 100]);
            
            % 算法选择
            uilabel(algoGrid, 'Text', '算法类型:');
            obj.AlgoDropDown = uidropdown(algoGrid, 'Items', {'Conventional', 'MUSIC', 'Functional'}, ...
                'Value', 'Conventional');
            
            % 算法详细参数
            paramGrid = uigridlayout(algoGrid, [1, 4]);
            paramGrid.Layout.Column = [1 2];
            paramGrid.Padding = [0 0 0 0];
            
            uilabel(paramGrid, 'Text', 'MUSIC源数:');
            obj.EditMusicN = uieditfield(paramGrid, 'numeric', 'Value', 1, 'Limits', [1 100], 'RoundFractionalValues', 'on');
            
            uilabel(paramGrid, 'Text', 'Gamma:');
            obj.EditFuncGamma = uieditfield(paramGrid, 'numeric', 'Value', 10, 'Limits', [1 1000]);
            
            runBtn = uibutton(algoGrid, 'Text', '运行波束成形', ...
                'BackgroundColor', [0.3, 0.6, 1.0], ...
                'FontWeight', 'bold', ...
                'ButtonPushedFcn', @obj.runBeamforming);
            runBtn.Layout.Column = [1 2];
            
            % 状态栏
            obj.StatusLabel = uilabel(vbox, 'Text', '就绪 | 提示: 点击3D图获取焦点后使用WASD移动');
            obj.StatusLabel.Layout.Row = 7;
            obj.StatusLabel.FontColor = [0.4 0.4 0.4];
            
            % === 右侧绘图区 ===
            obj.RightTabs = uitabgroup(obj.GridLayout);
            obj.RightTabs.Layout.Row = 1;
            obj.RightTabs.Layout.Column = 2;
            
            % Tab 1: 3D 视图
            obj.Tab3D = uitab(obj.RightTabs, 'Title', '3D 设置视图');
            gl3d = uigridlayout(obj.Tab3D, [1,1]);
            obj.Ax3D = uiaxes(gl3d);
            xlabel(obj.Ax3D, 'X (m)'); ylabel(obj.Ax3D, 'Y (m)'); zlabel(obj.Ax3D, 'Z (m)');
            grid(obj.Ax3D, 'on');
            view(obj.Ax3D, obj.ViewAzim, obj.ViewElev);
            
            % Tab 2: 波束成形结果
            obj.TabBF = uitab(obj.RightTabs, 'Title', '波束成形热力图');
            glbf = uigridlayout(obj.TabBF, [2, 1]);
            glbf.RowHeight = {'2x', '1x'};
            obj.AxBF_Map = uiaxes(glbf);
            title(obj.AxBF_Map, '波束成形结果');
            axis(obj.AxBF_Map, 'equal');
            
            obj.AxBF_Array = uiaxes(glbf);
            title(obj.AxBF_Array, '麦克风布局');
        end
        
        % === 交互回调函数 ===
        
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
                % Reset defaults
                obj.EditZFixed.Value = -0.3;
                obj.EditZStart.Value = -0.1;
                obj.EditZEnd.Value = -0.5;
                return;
            end

            z_coords = obj.Sources(:, 3);
            z_mean = mean(z_coords);
            z_min = min(z_coords);
            z_max = max(z_coords);

            % 设置固定深度 (平均值)
            obj.EditZFixed.Value = round(z_mean, 4);

            % 设置范围 (向外扩充 5cm)
            % 假设坐标系 Z 为负值，start 通常靠近 0，end 较深
            obj.EditZStart.Value = round(z_max + 0.05, 3);
            obj.EditZEnd.Value = round(z_min - 0.05, 3);
            
            % 自动切换模式：如果声源 Z 跨度超过 1cm，则切换到范围扫描
            if (z_max - z_min) > 0.01
                obj.ModeSwitch.Value = '范围(MIP)';
            else
                obj.ModeSwitch.Value = '固定深度';
            end
            
            % 刷新 UI 状态
            obj.onModeChanged(obj.ModeSwitch, []);
            
            obj.StatusLabel.Text = sprintf('参数已更新: 平均深度 %.3fm', z_mean);
        end
        
        function onAddSource(obj, ~, ~)
            newSrc = [obj.EditX.Value, obj.EditY.Value, obj.EditZ.Value];
            obj.Sources = [obj.Sources; newSrc];
            obj.updateSourceList();
            obj.plot3DScene();
            obj.StatusLabel.Text = sprintf('已添加声源: (%.2f, %.2f, %.2f)', newSrc);
            
            % 自动更新 MUSIC 源数推测值
            if obj.EditMusicN.Value < size(obj.Sources, 1)
                obj.EditMusicN.Value = size(obj.Sources, 1);
            end
        end
        
        function onRemoveSource(obj, ~, ~)
            idx = obj.SourceListBox.Value;
            if isempty(idx), return; end
            
            % 解析索引 (Listbox value 可能是字符串或数字，这里做简化处理)
            % MATLAB Listbox 如果 Items 是字符串数组，Value 返回选中的字符串
            % 我们需要维护索引
            if isempty(obj.SelectedSource), return; end
            
            obj.Sources(obj.SelectedSource, :) = [];
            obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene();
        end
        
        function onClearSources(obj, ~, ~)
            obj.Sources = [];
            obj.SelectedSource = [];
            obj.updateSourceList();
            obj.plot3DScene();
        end
        
        function onSourceSelect(obj, src, ~)
            idx = find(strcmp(src.Items, src.Value));
            if ~isempty(idx)
                obj.SelectedSource = idx;
                s = obj.Sources(idx, :);
                obj.EditX.Value = s(1);
                obj.EditY.Value = s(2);
                obj.EditZ.Value = s(3);
                obj.plot3DScene(); % 高亮显示
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
            
            % 恢复选中状态
            if ~isempty(obj.SelectedSource) && obj.SelectedSource <= n
                obj.SourceListBox.Value = items{obj.SelectedSource};
            end
        end
        
        function onKeyPress(obj, ~, event)
            % 处理 WASD / QE 移动声源
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
            
            % 更新坐标
            obj.Sources(obj.SelectedSource, :) = obj.Sources(obj.SelectedSource, :) + [dx, dy, dz];
            
            % 更新 UI 显示
            s = obj.Sources(obj.SelectedSource, :);
            obj.EditX.Value = s(1);
            obj.EditY.Value = s(2);
            obj.EditZ.Value = s(3);
            
            obj.updateSourceList();
            obj.plot3DScene();
        end
        
        % === 绘图逻辑 ===
        
        function plot3DScene(obj)
            ax = obj.Ax3D;
            % 保存视角
            [az, el] = view(ax);
            
            cla(ax);
            hold(ax, 'on');
            
            % 1. 绘制麦克风
            mx = obj.MicCoords(1,:);
            my = obj.MicCoords(2,:);
            mz = obj.MicCoords(3,:);
            
            scatter3(ax, mx, my, mz, 50, 'filled', 'MarkerFaceColor', [0 0.4470 0.7410]);
            
            % 绘制圆盘底座示意
            theta = 0:0.1:2*pi;
            max_r = max(sqrt(mx.^2 + my.^2)) * 1.1;
            fill3(ax, max_r*cos(theta), max_r*sin(theta), zeros(size(theta)), ...
                [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'k');
            
            % 2. 绘制声源
            nSrc = size(obj.Sources, 1);
            if nSrc > 0
                sx = obj.Sources(:,1);
                sy = obj.Sources(:,2);
                sz = obj.Sources(:,3);
                
                % 绘制投影线
                for i = 1:nSrc
                    plot3(ax, [sx(i) sx(i)], [sy(i) sy(i)], [sz(i) 0], 'r--', 'LineWidth', 1);
                end
                
                % 绘制声源点
                colors = repmat([1 0 0], nSrc, 1);
                sizes = repmat(100, nSrc, 1);
                
                if ~isempty(obj.SelectedSource)
                    colors(obj.SelectedSource, :) = [1 0.6 0]; % 橙色高亮
                    sizes(obj.SelectedSource) = 200;
                end
                
                scatter3(ax, sx, sy, sz, sizes, colors, 'filled', 'p'); % 五角星
            end
            
            % 调整坐标轴
            axis(ax, 'equal');
            % 确保Z轴即便没有声源也能看到负轴
            zlim(ax, [-0.6 0.1]); 
            
            view(ax, az, el);
            hold(ax, 'off');
        end
        
        % === 核心算法：波束成形 ===
        
        function runBeamforming(obj, ~, ~)
            if isempty(obj.Sources)
                uialert(obj.UIFigure, '请先添加声源!', '错误');
                return;
            end
            
            % 显示进度
            d = uiprogressdlg(obj.UIFigure, 'Title', '计算中', 'Message', '正在计算 CSM 及特征分解...', 'Indeterminate', 'on');
            
            try
                % 1. 参数准备
                freq = obj.EditFreq.Value;
                omega = 2 * pi * freq;
                k = omega / obj.c;
                algo = obj.AlgoDropDown.Value;
                snr = obj.EditSNR.Value; % 获取 SNR
                
                % 2. 计算互谱矩阵 (CSM)
                n_mics = size(obj.MicCoords, 2);
                n_src = size(obj.Sources, 1);
                CSM = zeros(n_mics, n_mics);
                
                % 计算理论信号 CSM
                for i = 1:n_src
                    src_pos = obj.Sources(i,:)';
                    vec = obj.MicCoords - src_pos;
                    dist = sqrt(sum(vec.^2, 1));
                    signal_vec = (1./dist) .* exp(-1j * k * dist);
                    CSM = CSM + (signal_vec.' * conj(signal_vec)); 
                end
                
                % 3. 添加噪声 (模拟传感器噪声/对角加载)
                avg_signal_power = trace(CSM) / n_mics;
                
                if avg_signal_power > 0
                    % 根据 SNR 计算噪声功率: P_noise = P_signal / 10^(SNR/10)
                    noise_power = avg_signal_power / (10^(snr/10));
                else
                    noise_power = 1e-12; % 极小值防止全零
                end
                
                % 添加对角加载 (模拟非相关噪声)
                CSM = CSM + noise_power * eye(n_mics);
                
                % 4. 算法预处理 (准备 M_core 矩阵)
                % Bartlett: M_core = CSM; Power = w' * M_core * w
                % MUSIC: M_core = Un*Un'; Power = 1 / (w' * M_core * w)
                % Functional: M_core = CSM^(1/gamma); Power = (w' * M_core * w)^gamma
                
                M_core = CSM;
                func_gamma = obj.EditFuncGamma.Value;
                
                if strcmp(algo, 'MUSIC') || strcmp(algo, 'Functional')
                    [V, D] = eig(CSM);
                    [lambda, idx] = sort(diag(real(D)), 'descend');
                    V = V(:, idx);
                    
                    if strcmp(algo, 'MUSIC')
                        % 子空间分解
                        n_music = min(obj.EditMusicN.Value, n_mics-1);
                        Un = V(:, n_music+1:end); % 噪声子空间
                        M_core = Un * Un';
                    elseif strcmp(algo, 'Functional')
                        % 功能性波束成形 (指数加权)
                        lambda = max(lambda, 0); % 避免数值误差导致的负特征值
                        D_nu = diag(lambda .^ (1/func_gamma));
                        M_core = V * D_nu * V';
                    end
                end
                
                % 5. 定义扫描网格 (XY平面)
                grid_res = 0.005; % 5mm 分辨率
                x_vec = -0.2:grid_res:0.2;
                y_vec = -0.2:grid_res:0.2;
                [GridX, GridY] = meshgrid(x_vec, y_vec);
                [rows, cols] = size(GridX);
                
                % 6. 波束成形计算
                mode = obj.ModeSwitch.Value;
                model = obj.ModelDropDown.Value;
                
                final_map = zeros(rows, cols);
                
                if strcmp(mode, '固定深度')
                    z_target = obj.EditZFixed.Value;
                    d.Message = sprintf('正在计算 Z=%.3f 平面 (%s)...', z_target, algo);
                    
                    GridZ = ones(size(GridX)) * z_target;
                    final_map = obj.computeLayer(M_core, GridX, GridY, GridZ, k, model, algo, func_gamma);
                    
                else % 范围扫描 (MIP)
                    z_start = obj.EditZStart.Value;
                    z_end = obj.EditZEnd.Value;
                    z_step = obj.EditZStep.Value;
                    
                    if z_start > z_end, temp=z_start; z_start=z_end; z_end=temp; end
                    z_layers = z_start:z_step:z_end;
                    if isempty(z_layers), z_layers = z_start; end
                    
                    d.Indeterminate = 'off';
                    
                    for idx = 1:length(z_layers)
                        z_curr = z_layers(idx);
                        d.Value = idx / length(z_layers);
                        d.Message = sprintf('扫描深度 Z=%.3f (%d/%d)...', z_curr, idx, length(z_layers));
                        
                        GridZ = ones(size(GridX)) * z_curr;
                        layer_map = obj.computeLayer(M_core, GridX, GridY, GridZ, k, model, algo, func_gamma);
                        
                        final_map = max(final_map, layer_map);
                    end
                end
                
                % 转换为 dB
                final_map_dB = 10 * log10(abs(final_map) + 1e-12);
                max_val = max(final_map_dB(:));
                
                % 7. 绘图
                obj.plotResults(GridX, GridY, final_map_dB, max_val);
                
                obj.StatusLabel.Text = sprintf('%s 计算完成', algo);
                close(d);
                
                obj.RightTabs.SelectedTab = obj.TabBF;
                
            catch ME
                close(d);
                uialert(obj.UIFigure, ME.message, '计算错误');
            end
        end
        
        function bf_map = computeLayer(obj, M_core, GX, GY, GZ, k, model, algo, gamma)
            % 通用计算层：根据传入的 M_core 和算法类型计算能量分布
            
            [rows, cols] = size(GX);
            n_points = numel(GX);
            n_mics = size(obj.MicCoords, 2);
            
            grid_pts = [GX(:)'; GY(:)'; GZ(:)'];
            power_vec = zeros(1, n_points);
            
            mic_x = obj.MicCoords(1,:);
            mic_y = obj.MicCoords(2,:);
            mic_z = obj.MicCoords(3,:);
            
            % 并行计算或矢量化计算在 MATLAB 中通常很快
            % 这里保持显式循环以确保逻辑清晰，并便于内存管理
            for i = 1:n_points
                focus_pt = grid_pts(:, i);
                
                % 计算距离和导向向量 w
                dx = focus_pt(1) - mic_x;
                dy = focus_pt(2) - mic_y;
                dz = focus_pt(3) - mic_z;
                r = sqrt(dx.^2 + dy.^2 + dz.^2);
                
                if strcmp(model, 'near')
                    % 近场球面波
                    w = (1./r) .* exp(-1j * k * r);
                    w = w / norm(w); 
                else
                    % 远场平面波
                    u = focus_pt / norm(focus_pt);
                    phase = k * (mic_x*u(1) + mic_y*u(2) + mic_z*u(3));
                    w = exp(-1j * phase);
                    w = w / n_mics;
                end
                w = w(:);
                
                % 核心功率计算
                % val = w' * M_core * w
                val = real(w' * M_core * w);
                
                if strcmp(algo, 'MUSIC')
                    % MUSIC: P = 1 / (w' * Un * Un' * w)
                    % 避免除以零
                    power_vec(i) = 1 / (val + 1e-15);
                elseif strcmp(algo, 'Functional')
                    % Functional: P = (w' * C^(1/nu) * w)^nu
                    power_vec(i) = val ^ gamma;
                else
                    % Conventional: P = w' * C * w
                    power_vec(i) = val;
                end
            end
            
            bf_map = reshape(power_vec, rows, cols);
        end
        
        function plotResults(obj, GridX, GridY, map_dB, max_dB)
            % 绘制热力图
            ax = obj.AxBF_Map;
            cla(ax);
            hold(ax, 'on');
            
            % 动态范围调整
            algo = obj.AlgoDropDown.Value;
            if strcmp(algo, 'MUSIC')
                dr = 40; % MUSIC 动态范围较大
            elseif strcmp(algo, 'Functional')
                dr = 30; % Functional 动态范围也较大
            else
                dr = 15; % 常规波束成形
            end
            
            clims = [max_dB - dr, max_dB];
            imagesc(ax, GridX(1,:), GridY(:,1), map_dB, clims);
            colormap(ax, 'jet');
            
            % 更新 Colorbar Label
            c = colorbar(ax);
            c.Label.String = sprintf('SPL [dB] (%s)', algo);
            
            axis(ax, 'xy');
            
            % 绘制真实声源
            if obj.ShowStarsCheck.Value
                sx = obj.Sources(:,1);
                sy = obj.Sources(:,2);
                plot(ax, sx, sy, 'w*', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerEdgeColor', 'k');
            end
            
            hold(ax, 'off');
            
            % 绘制下方阵列图
            ax2 = obj.AxBF_Array;
            cla(ax2);
            plot(ax2, obj.MicCoords(1,:), obj.MicCoords(2,:), 'o');
            grid(ax2, 'on');
            axis(ax2, 'equal');
            title(ax2, '阵列 XY 投影');
        end
    end
end