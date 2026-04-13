# DaVinci Resolve 脚本集成原理讲解

## 概述

这个项目通过 DaVinci Resolve 的官方脚本 API 和 Fusion UI 框架，实现了一个完整的 TTS（文字转语音）插件。让我们深入了解它是如何工作的。

---

## 一、DaVinci Resolve 脚本系统架构

### 1.1 脚本放置位置

DaVinci Resolve 会自动扫描特定目录下的 Python 脚本：

**Mac:**
```
/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Edit
```

**Windows:**
```
C:\ProgramData\Blackmagic Design\DaVinci Resolve\Fusion\Scripts\Edit
```

**关键点：**
- 放在 `Scripts/Edit` 目录下的脚本会出现在 DaVinci Resolve 的 **工作区 → 脚本** 菜单中
- 用户点击菜单项时，DaVinci Resolve 会执行对应的 Python 脚本
- 脚本在 DaVinci Resolve 的 Python 环境中运行，可以访问其内置的 API

---

## 二、核心集成机制

### 2.1 DaVinci Resolve API 导入

```python
# ================== DaVinci Resolve 接入 ==================
try:
    import DaVinciResolveScript as dvr_script
    from python_get_resolve import GetResolve
    print("DaVinciResolveScript from Python")
except ImportError:
    # 如果导入失败，手动添加 DaVinci Resolve 的脚本路径
    if platform.system() == "Darwin":  # Mac
        path1 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Examples"
        path2 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
    elif platform.system() == "Windows":
        path1 = os.path.join(os.environ['PROGRAMDATA'], "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Examples")
        path2 = os.path.join(os.environ['PROGRAMDATA'], "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Modules")
    
    sys.path += [path1, path2]
    import DaVinciResolveScript as dvr_script
    from python_get_resolve import GetResolve
```

**工作原理：**
1. **DaVinciResolveScript** - DaVinci Resolve 的官方 Python API 模块
2. **GetResolve** - 获取 DaVinci Resolve 应用实例的函数
3. 如果导入失败，脚本会自动添加 DaVinci Resolve 的脚本路径到 `sys.path`

### 2.2 获取 DaVinci Resolve 对象

```python
def connect_resolve():
    # resolve = dvr_script.scriptapp("Resolve")  # 另一种方式
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    return resolve, project, timeline

resolve, current_project, current_timeline = connect_resolve()
```

**对象层级：**
```
Resolve (应用)
  └─ ProjectManager (项目管理器)
       └─ Project (当前项目)
            └─ Timeline (当前时间线)
                 ├─ MediaPool (媒体池)
                 ├─ Tracks (轨道)
                 └─ TimelineItems (时间线项目)
```

---

## 三、Fusion UI 界面系统

### 3.1 UI 管理器初始化

```python
# 这两个对象是 DaVinci Resolve 内置的全局对象
ui = fusion.UIManager          # UI 管理器
dispatcher = bmd.UIDispatcher(ui)  # 事件分发器
```

**关键概念：**
- **fusion** - DaVinci Resolve 内置的 Fusion 模块（全局对象）
- **bmd** - Blackmagic Design 的缩写，提供底层 UI 功能
- **UIManager** - 负责创建和管理 UI 组件
- **UIDispatcher** - 负责处理 UI 事件（按钮点击、输入等）

### 3.2 创建窗口

```python
loading_win = dispatcher.AddWindow(
    {
        "ID": "LoadingWin",                    # 窗口唯一标识
        "WindowTitle": "Loading",              # 窗口标题
        "Geometry": [X_CENTER, Y_CENTER, WINDOW_WIDTH, WINDOW_HEIGHT],  # [x, y, width, height]
        "Spacing": 10,                         # 组件间距
        "StyleSheet": "*{font-size:14px;}"     # CSS 样式
    },
    [
        ui.VGroup(                             # 垂直布局组
            [
                ui.Label(                      # 标签组件
                    {
                        "ID": "LoadLabel", 
                        "Text": "Loading...",
                        "Alignment": {"AlignHCenter": True, "AlignVCenter": True},
                    }
                )
            ]
        )
    ]
)
loading_win.Show()  # 显示窗口
```

**UI 组件类型：**
- **VGroup / HGroup** - 垂直/水平布局容器
- **Label** - 文本标签
- **Button** - 按钮
- **LineEdit** - 单行文本输入框
- **TextEdit** - 多行文本编辑器
- **ComboBox** - 下拉选择框
- **Slider** - 滑块
- **CheckBox** - 复选框
- **TabBar** - 标签页

### 3.3 事件处理机制

```python
# 获取窗口中的所有组件
items = main_window.GetItems()

# 定义事件处理函数
def OnButtonClick(ev):
    button_id = ev['who']  # 获取触发事件的组件 ID
    
    if button_id == "SynthesizeButton":
        # 处理合成按钮点击
        text = items["TextEdit"].PlainText  # 获取文本框内容
        # ... 执行 TTS 合成
    
    elif button_id == "GetSubtitleButton":
        # 处理获取字幕按钮点击
        # ... 从时间线提取字幕

# 注册事件监听器
main_window.On.SynthesizeButton.Clicked = OnButtonClick
main_window.On.GetSubtitleButton.Clicked = OnButtonClick

# 启动事件循环
dispatcher.RunLoop()  # 阻塞，等待用户交互
```

**事件流程：**
1. 用户点击按钮
2. Fusion UI 触发事件
3. 调用注册的回调函数
4. 回调函数执行业务逻辑
5. 更新 UI 显示

---

## 四、与 DaVinci Resolve 交互

### 4.1 从时间线获取字幕

```python
def get_subtitle_from_timeline():
    """从当前播放头位置获取字幕"""
    timeline = current_project.GetCurrentTimeline()
    current_frame = timeline.GetCurrentTimeCode()  # 获取当前帧
    
    # 遍历所有轨道
    for track_index in range(1, 100):
        items = timeline.GetItemListInTrack("subtitle", track_index)
        
        for item in items:
            start_frame = item.GetStart()
            end_frame = item.GetEnd()
            
            # 检查播放头是否在字幕范围内
            if start_frame <= current_frame <= end_frame:
                subtitle_text = item.GetName()  # 获取字幕文本
                return subtitle_text
    
    return None
```

### 4.2 添加音频到媒体池和时间线

```python
def add_to_media_pool_and_timeline(start_frame, end_frame, audio_file_path):
    """将音频添加到媒体池并插入时间线"""
    
    # 1. 获取媒体池
    media_pool = current_project.GetMediaPool()
    
    # 2. 导入音频文件到媒体池
    media_pool_item = media_pool.ImportMedia([audio_file_path])[0]
    
    # 3. 获取当前时间线
    timeline = current_project.GetCurrentTimeline()
    
    # 4. 找到第一个空的音频轨道
    track_index = get_first_empty_track(timeline, start_frame, end_frame, "audio")
    
    # 5. 将音频添加到时间线
    timeline.InsertMediaPoolItem(
        media_pool_item,      # 媒体池项目
        track_index,          # 轨道索引
        start_frame           # 起始帧
    )
```

### 4.3 渲染音频（用于音色复刻）

```python
def render_audio_by_marker(output_dir):
    """根据 Marker 标记渲染音频片段"""
    
    # 1. 切换到单一剪辑渲染模式
    current_project.SetCurrentRenderMode(1)
    
    # 2. 获取时间线的 Marker
    markers = current_timeline.GetMarkers()
    first_marker = sorted(markers.keys())[0]
    marker_info = markers[first_marker]
    
    # 3. 设置渲染参数
    render_settings = {
        "SelectAllFrames": False,
        "MarkIn": start_frame,
        "MarkOut": end_frame,
        "TargetDir": output_dir,
        "CustomName": filename,
        "ExportVideo": False,
        "ExportAudio": True,
        "AudioCodec": "LinearPCM",
        "AudioBitDepth": 16,
        "AudioSampleRate": 48000,
    }
    
    # 4. 加载音频预设
    current_project.LoadRenderPreset("Audio Only")
    
    # 5. 设置渲染参数并开始渲染
    current_project.SetRenderSettings(render_settings)
    job_id = current_project.AddRenderJob()
    current_project.StartRendering([job_id], isInteractiveMode=False)
    
    # 6. 等待渲染完成
    while current_project.IsRenderingInProgress():
        time.sleep(0.5)
```

---

## 五、完整的工作流程

### 5.1 启动流程

```
用户点击菜单 "工作区 → 脚本 → DaVinci TTS"
    ↓
DaVinci Resolve 执行 Python 脚本
    ↓
脚本导入 DaVinciResolveScript 和 Fusion UI
    ↓
显示加载窗口
    ↓
加载依赖库（requests, azure-speech, edge-tts）
    ↓
创建主窗口界面
    ↓
进入事件循环，等待用户操作
```

### 5.2 TTS 合成流程

```
用户点击"朗读文本框"按钮
    ↓
获取文本框内容
    ↓
调用 TTS API（Azure/MiniMax/Edge TTS）
    ↓
接收音频数据
    ↓
保存音频文件到本地
    ↓
调用 DaVinci Resolve API：
    - 导入音频到媒体池
    - 插入音频到时间线
    ↓
更新 UI 状态
```

### 5.3 音色复刻流程

```
用户在时间线上添加 Marker 标记音频范围
    ↓
点击"上传音频"按钮
    ↓
调用 DaVinci Resolve 渲染 API
    ↓
渲染指定范围的音频
    ↓
上传音频到 MiniMax API
    ↓
获取 file_id
    ↓
调用音色复刻 API
    ↓
保存复刻的 voice_id
```

---

## 六、关键技术点

### 6.1 全局对象

DaVinci Resolve 在脚本运行时提供了几个全局对象：

```python
# 这些对象无需导入，直接可用
fusion      # Fusion 模块，提供 UI 功能
bmd         # Blackmagic Design 模块
resolve     # DaVinci Resolve 应用实例（需要通过 GetResolve() 获取）
```

### 6.2 线程安全

```python
# UI 更新必须在主线程中进行
def update_ui_safely(text):
    # 使用 QTimer 或直接更新
    items["StatusLabel"].Text = text
    
# 后台任务使用线程
def background_task():
    # 执行耗时操作（API 调用、文件 I/O）
    result = call_tts_api(text)
    
    # 更新 UI（回到主线程）
    update_ui_safely(f"完成：{result}")

thread = threading.Thread(target=background_task)
thread.start()
```

### 6.3 配置持久化

```python
# 保存配置到 JSON 文件
def save_settings(settings, settings_file):
    with open(settings_file, 'w') as file:
        json.dump(settings, file, indent=4)

# 加载配置
def load_settings(settings_file):
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as file:
            return json.load(file)
    return DEFAULT_SETTINGS
```

配置文件位置：
```
DaVinci Resolve TTS/config/TTS_settings.json
```

---

## 七、UI 布局示例

### 7.1 主窗口结构

```python
main_window = dispatcher.AddWindow(
    {...},
    [
        ui.VGroup([
            # 标签页
            ui.TabBar({"ID": "MainTabs"}),
            
            # 标签页内容栈
            ui.Stack({"ID": "ContentStack"}, [
                # Azure TTS 页面
                ui.VGroup([
                    ui.HGroup([
                        ui.Label({"Text": "文本:"}),
                        ui.TextEdit({"ID": "AzureTextEdit"}),
                    ]),
                    ui.HGroup([
                        ui.Button({"ID": "AzureSynthesizeBtn", "Text": "合成"}),
                        ui.Button({"ID": "AzurePreviewBtn", "Text": "预览"}),
                    ]),
                ]),
                
                # MiniMax TTS 页面
                ui.VGroup([
                    # ... 类似结构
                ]),
                
                # 配置页面
                ui.VGroup([
                    # ... 配置选项
                ]),
            ]),
        ])
    ]
)
```

### 7.2 动态更新 UI

```python
# 切换标签页
def on_tab_changed(ev):
    tab_index = ev['Index']
    items["ContentStack"].CurrentIndex = tab_index

# 更新下拉框选项
def update_voice_list(voices):
    items["VoiceComboBox"].Clear()
    for voice in voices:
        items["VoiceComboBox"].AddItem(voice)

# 显示/隐藏组件
items["AdvancedSettings"].Visible = False
items["AdvancedSettings"].Visible = True
```

---

## 八、调试技巧

### 8.1 查看日志

```python
# 打印到 DaVinci Resolve 控制台
print("Debug info:", variable)

# 显示消息框
def show_message(title, message):
    msg_win = dispatcher.AddWindow(
        {"WindowTitle": title},
        [ui.Label({"Text": message})]
    )
    msg_win.Show()
```

### 8.2 错误处理

```python
try:
    # 执行 DaVinci Resolve API 调用
    timeline = current_project.GetCurrentTimeline()
except Exception as e:
    print(f"Error: {e}")
    show_message("错误", str(e))
```

---

## 九、总结

### DaVinci Resolve 脚本集成的核心要素：

1. **脚本放置位置** - 特定目录下的 `.py` 文件会自动出现在菜单中
2. **API 导入** - 通过 `DaVinciResolveScript` 访问 DaVinci Resolve 功能
3. **Fusion UI** - 使用 `fusion.UIManager` 创建图形界面
4. **事件驱动** - 通过 `dispatcher.RunLoop()` 处理用户交互
5. **对象层级** - Resolve → Project → Timeline → MediaPool/Tracks
6. **全局对象** - `fusion`, `bmd`, `resolve` 在脚本中直接可用

### 优势：

- ✅ 无需编译，纯 Python 脚本
- ✅ 直接访问 DaVinci Resolve 的所有功能
- ✅ 可以创建复杂的图形界面
- ✅ 支持跨平台（Mac/Windows/Linux）
- ✅ 可以调用外部 API 和库

### 局限性：

- ⚠️ 必须在 DaVinci Resolve 环境中运行
- ⚠️ UI 框架相对简单，不如 Qt/Tkinter 丰富
- ⚠️ 文档较少，需要参考官方示例

---

## 参考资源

- [DaVinci Resolve Scripting Documentation](https://documents.blackmagicdesign.com/DeveloperManuals/DaVinci_Resolve_Scripting_API.pdf)
- [Fusion Scripting Guide](https://documents.blackmagicdesign.com/UserManuals/Fusion8_Scripting_Guide.pdf)
- 官方示例脚本位置：
  - Mac: `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Examples`
  - Windows: `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Examples`

---

**文档生成时间：** 2025-11-30
