# P1 剩余功能设计：音频保留、图片输入、拖拽支持

**日期**: 2026-04-04
**基于**: DESIGN.md 3.1.1, 3.1.2, 3.1.4

---

## 1. 音频保留

### 目标

视频输出时保留原始音轨，音频直接 copy 不重新编码。

### 设计

**修改 `ProResWriter` 和 `FFmpegWriter`**：构造函数新增可选参数 `audio_source: str | None = None`。

当 `audio_source` 不为 None 时，FFmpeg 命令变为：

```
ffmpeg -y -f rawvideo -pix_fmt ... -s WxH -r FPS -i - \
       -i {audio_source} \
       -map 0:v -map 1:a? -c:a copy \
       -shortest \
       {其他编码参数} output_path
```

关键点：
- `-map 1:a?` 中的 `?` 表示如果原视频没有音轨则忽略（不报错）
- `-shortest` 确保视频和音频对齐
- `-c:a copy` 直接拷贝音轨，无需重编码
- 无需先用 ffprobe 检测音轨——FFmpeg 的 `?` 映射自动处理

**修改 `create_writer()`**：新增 `audio_source` 参数，透传给 FFmpeg 类 writer。`ImageSequenceWriter` 忽略此参数。

**修改 `MattingPipeline.process()`**：将 `input_path` 作为 `audio_source` 传给 `create_writer()`。

**UI**：无变化。默认保留音频，无开关。

**不涉及的场景**：
- 图片序列输出（`ImageSequenceWriter`）：天然无音频，忽略
- 图片输入：不涉及音频

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/core/video.py` | `ProResWriter.__init__` 新增 `audio_source` 参数，修改 FFmpeg 命令 |
| `src/core/writer.py` | `FFmpegWriter.__init__` 新增 `audio_source` 参数；`create_writer()` 新增 `audio_source` 参数并透传 |
| `src/core/pipeline.py` | `process()` 中调用 `create_writer()` 时传入 `input_path` 作为 `audio_source` |

---

## 2. 图片输入

### 目标

支持单张图片和图片文件夹作为输入，复用现有的推理和合成逻辑，输出固定为 PNG。

### 输入类型识别

通过文件扩展名区分：
- **视频**：`.mp4`, `.avi`, `.mov`, `.mkv`
- **图片**：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`
- **文件夹**：`os.path.isdir()` 判断，扫描内部图片文件

### 设计

**新增 `src/core/image_pipeline.py`**：

```python
class ImagePipeline:
    def __init__(self, config: ProcessingConfig, models_dir: str):
        # 加载模型（同 MattingPipeline）

    def process(self, input_path: str, output_dir: str,
                pause_event, cancel_event, progress_callback) -> str:
        """处理单张图片或图片文件夹。

        input_path: 图片文件路径或包含图片的文件夹路径
        output_dir: 输出目录
        返回: 输出路径（单图返回文件路径，文件夹返回目录路径）
        """
```

**单张图片处理流程**：
1. `cv2.imread()` 读取 BGR 帧
2. `predict()` 获取 alpha mask
3. `compose_frame()` 合成
4. PIL 保存为 PNG（透明模式 RGBA，其他模式 RGB）
5. 输出命名：`{原文件名}_{model}_{毫秒时间戳}.png`

**文件夹批量处理流程**：
1. 扫描文件夹中所有图片文件（支持嵌套？不支持，仅顶层）
2. 按文件名排序
3. 逐张处理，每张走单张流程
4. 输出到 `output_dir/` 下，保持原文件名（加后缀）
5. 进度回调：`progress_callback(current_index, total_count)`
6. 支持暂停/取消

### GUI 变更

**文件选择对话框**：
- 新增"选择图片"和"选择文件夹"按钮，或统一为一个按钮弹出菜单
- 方案：改为三个按钮：`选择视频`、`选择图片`、`选择文件夹`
- 或者：一个 `选择文件` 按钮，过滤器包含视频和图片格式，加一个独立的 `选择文件夹` 按钮

**推荐方案**：保持一个 `选择文件` 按钮（过滤器同时包含视频和图片），加一个 `选择文件夹` 按钮。根据选中的文件类型自动切换模式。

**图片模式下的 UI 调整**：
- 输出格式下拉框：禁用（固定 PNG 输出）
- 背景模式：保持可选（透明/绿幕/蓝幕/蒙版等）
- 视频信息区域：显示图片信息（尺寸、文件数）
- 输出路径：图片模式下选择输出目录（而非文件）

**Worker 变更**：
- `MattingWorker.run()` 根据输入类型分支：
  - 视频 → `MattingPipeline`
  - 图片/文件夹 → `ImagePipeline`

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/core/image_pipeline.py` | **新增**：ImagePipeline 类 |
| `src/gui/main_window.py` | 文件选择扩展；输入类型检测；图片模式 UI 适配 |
| `src/worker/matting_worker.py` | 根据输入类型选择 pipeline |
| `src/core/config.py` | 新增 `InputType` 枚举（VIDEO / IMAGE / IMAGE_FOLDER） |

---

## 3. 拖拽支持

### 目标

支持将视频文件、图片文件、图片文件夹直接拖拽到主窗口。

### 设计

**修改 `MainWindow`**：

```python
def __init__(self):
    ...
    self.setAcceptDrops(True)

def dragEnterEvent(self, event: QDragEnterEvent):
    if event.mimeData().hasUrls():
        event.acceptProposedAction()

def dropEvent(self, event: QDropEvent):
    urls = event.mimeData().urls()
    if not urls:
        return
    path = urls[0].toLocalFile()  # 只取第一个文件
    self._handle_input(path)
```

**`_handle_input(path)`** 统一入口：
1. 判断 path 是文件还是目录
2. 文件：根据扩展名判断视频/图片
3. 目录：扫描是否包含图片文件
4. 调用与文件选择对话框相同的后续逻辑（显示信息、切换 UI 模式、设为 ready 状态）
5. 不支持的类型：弹出提示

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/gui/main_window.py` | 新增 `dragEnterEvent`、`dropEvent`、`_handle_input` 方法；重构 `_on_select_file` 共用 `_handle_input` |

---

## 4. 实现顺序

1. **音频保留** — 改动最小，只涉及 writer 层
2. **图片输入** — 新增 pipeline + GUI 适配
3. **拖拽支持** — 依赖图片输入完成后的 `_handle_input` 统一入口

---

## 5. 测试计划

| 功能 | 测试点 |
|------|--------|
| 音频保留 | 有音轨视频→输出包含音频；无音轨视频→正常输出无报错；图片序列输出→不受影响 |
| 图片输入 | 单张 PNG→输出 PNG；单张 JPG→输出 PNG；文件夹 5 张→输出 5 张；空文件夹→报错提示；暂停/取消→正常中断 |
| 拖拽支持 | 拖入视频→识别为视频模式；拖入图片→识别为图片模式；拖入文件夹→识别为文件夹模式；拖入 .txt→提示不支持 |
