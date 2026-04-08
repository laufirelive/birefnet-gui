# 单帧预览功能设计规格

## 1. 目标

在处理整个视频前，用户可以选择任意一帧进行抠图预览，确认模型和参数效果是否满意，避免跑完整个视频才发现问题。

## 2. 布局调整

### 2.1 当前左侧面板布局（自上而下）

```
输入文件 + 选择按钮
视频信息标签
─── 分隔线 ───
输出路径 + 浏览按钮
─── 分隔线 ───
进度条
状态标签
stretch
```

### 2.2 新布局（自上而下）

```
输入文件 + 选择按钮
视频信息标签
─── 分隔线 ───
输出路径 + 浏览按钮
─── 分隔线 ───
预览窗格 (QLabel, 缩放显示, stretch=1)
帧滑块 (QSlider horizontal)
帧信息标签 ("帧: 1620/5400  00:54.0")
─── 分隔线 ───
进度条
状态标签
```

关键变化：
- 进度条和状态标签下移到左侧面板最底部
- 中间新增预览窗格 + 帧滑块 + 帧信息标签
- 预览窗格占据左侧面板的主要可伸缩空间 (`stretch=1`)

### 2.3 底部操作栏

在现有按钮序列中新增「预览」按钮：

```
[预览]  [开始处理]  [暂停]  [取消]  [加入队列]
```

## 3. 预览窗格

### 3.1 组件

- **预览图片区**: `QLabel`，使用 `setPixmap()` 显示帧图片，`setScaledContents(False)`，图片按比例缩放适应区域（`Qt.AspectRatioMode.KeepAspectRatio`）
- **帧滑块**: `QSlider(Qt.Orientation.Horizontal)`，范围 0 ~ frame_count-1
- **帧信息标签**: `QLabel`，显示 "帧: N/总帧数  MM:SS.s"

### 3.2 初始状态

- 未选择文件时：预览窗格显示空白或浅灰占位文字（"选择视频后预览"）
- 滑块禁用，帧信息标签为空

## 4. 帧截取

### 4.1 方式

使用 FFmpeg subprocess 按时间戳截取单帧：

```
ffmpeg -ss {seconds} -i {input_path} -frames:v 1 -f image2pipe -vcodec rawvideo -pix_fmt rgb24 -
```

从 stdout 读取原始 RGB 像素数据，构建为 numpy 数组，再转为 QPixmap 显示。

### 4.2 时间计算

```python
time_seconds = frame_number / fps
```

### 4.3 触发时机

- **选择视频后**: 自动截取第 0 帧显示
- **滑块释放时** (`sliderReleased` signal): 截取滑块指向的帧并显示
- 拖动过程中不截取（避免频繁启动 ffmpeg 子进程）

### 4.4 封装

新增 `src/core/frame_extractor.py` 模块，提供：

```python
def extract_frame(input_path: str, frame_number: int, fps: float, width: int, height: int) -> np.ndarray:
    """用 FFmpeg 截取指定帧，返回 RGB uint8 数组 (H, W, 3)。"""
```

## 5. 预览推理

### 5.1 流程

用户点击底部「预览」按钮后：

1. 按钮禁用，状态标签显示 "正在加载模型..."
2. 在后台线程（`PreviewWorker(QThread)`）中：
   a. 加载模型（`load_model`）
   b. 对当前预览帧执行单帧推理（`predict`）
   c. 释放模型（`del model + gc.collect + torch.cuda.empty_cache`）
3. 推理完成后通过 signal 返回 alpha mask
4. 主线程将抠图结果以棋盘格背景渲染并显示在预览窗格
5. 按钮恢复可用

### 5.2 PreviewWorker

新增 `src/worker/preview_worker.py`：

```python
class PreviewWorker(QThread):
    finished = pyqtSignal(np.ndarray)  # alpha mask
    error = pyqtSignal(str)

    def __init__(self, model_name: str, models_dir: str, frame: np.ndarray, resolution: int):
        ...

    def run(self):
        device = detect_device()
        model = load_model(get_model_path(self.model_name, self.models_dir), device)
        try:
            alpha = predict(model, self.frame, device, self.resolution)
            self.finished.emit(alpha)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### 5.3 模型策略

**每次预览：加载 → 推理 → 释放。** 不缓存模型实例。

理由：简单可靠，不长期占用显存，单帧预览对 3-5 秒加载时间可接受。

## 6. 棋盘格背景渲染

抠图结果显示时，透明区域用灰白棋盘格表示。

### 6.1 实现

```python
def render_checkerboard_preview(rgb: np.ndarray, alpha: np.ndarray, cell_size: int = 16) -> np.ndarray:
    """将 RGB + alpha 合成到棋盘格背景上，返回 RGB 图像。"""
```

- 生成与图片同尺寸的棋盘格背景（交替的 #CCCCCC 和 #FFFFFF 方块）
- 用 alpha 通道将前景与棋盘格背景混合
- 返回 RGB 图像用于 QPixmap 显示

放置在 `src/core/compositing.py` 中（现有合成模块）。

## 7. 交互状态

### 7.1 预览按钮状态

| 应用状态 | 预览按钮 |
|---------|---------|
| 未选择文件 | 禁用 |
| 已选择视频/图片，空闲 | 可用 |
| 正在预览推理 | 禁用（显示加载状态） |
| 正在处理视频 | 禁用 |

### 7.2 滑块与预览的联动

- 拖动滑块释放后 → 截取新帧显示 → **清除之前的抠图结果**（回到显示原始帧）
- 点击预览按钮 → 对当前帧抠图 → 预览窗格切换为显示抠图结果
- 再次拖动滑块 → 回到显示原始帧，可选择新帧再次预览

### 7.3 图片输入

- 图片输入时：滑块隐藏，预览窗格直接显示该图片
- 预览按钮仍然可用，点击后执行抠图预览
- 图片文件夹输入时：预览功能不可用（多张图片无法单帧预览）

## 8. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/core/frame_extractor.py` | 新增 | FFmpeg 单帧截取 |
| `src/core/compositing.py` | 修改 | 新增 `render_checkerboard_preview` |
| `src/worker/preview_worker.py` | 新增 | 预览推理后台线程 |
| `src/gui/main_window.py` | 修改 | 布局调整 + 预览窗格 + 滑块 + 预览按钮 + 状态管理 |

## 9. 不做的事

- 不做左右对比 / 拖拽分割线对比
- 不做预览结果缓存（每次预览重新推理）
- 不做视频播放功能（只是帧截取预览）
- 不做批量队列中的预览
