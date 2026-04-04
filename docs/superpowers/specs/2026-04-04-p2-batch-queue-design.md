# P2 批量队列设计

**日期**: 2026-04-04
**基于**: DESIGN.md 3.1.5, 3.1.6

---

## 1. 概述

实现完整的批量队列功能，包括：多任务排队处理、独立设置、拖拽排序、断点续传、工程文件持久化、完成提示音。

### 核心用户场景

用户需要处理多个视频/图片，希望：
1. 在 Tab 1 选好文件和设置后，点"加入队列"
2. 重复添加多个任务
3. 切到 Tab 2，点"开始队列"一键处理所有任务
4. 中途可暂停/关闭软件，下次打开自动恢复队列，从断点继续

---

## 2. 架构设计

### 2.1 Tab 切换方案

MainWindow 的中央 widget 改为 `QTabWidget`，两个 Tab：

- **Tab 1「单任务」**：保留现有界面，新增「加入队列」按钮
- **Tab 2「批量队列」**：队列任务列表 + 队列控制

两个 Tab 互斥执行：
- 单任务在处理时，不能启动队列
- 队列在执行时，单任务 Tab 的「开始处理」禁用

### 2.2 Tab 1 改动

在控制按钮行中，「开始处理」旁边新增「加入队列」按钮。

点击「加入队列」后：
1. 将当前文件路径 + 输入类型 + ProcessingConfig + 输出目录打包为 QueueTask
2. 添加到 QueueManager
3. Tab 2 标题更新任务数
4. 清空 Tab 1 输入（方便继续添加下一个）
5. 弹出简短提示："已加入队列"

### 2.3 两阶段 Pipeline

所有处理（包括单任务模式）统一走两阶段流程：

```
阶段1 (推理): FrameReader → predict → save mask PNG → 更新进度
阶段2 (编码): FrameReader + mask PNGs → compose → FFmpegWriter
```

**推理阶段**：
- 逐帧读取原视频帧
- BiRefNet 推理生成 alpha mask
- mask 保存为 grayscale PNG 到缓存目录（~200KB/帧 @1080p）
- 进度回调：`推理中: {current}/{total}帧 | {fps} FPS | 剩余 {time}`

**编码阶段**：
- 逐帧读取原视频帧 + 对应缓存 mask
- compose 合成（透明/绿幕/蒙版等）
- FFmpegWriter 写入最终输出文件
- 进度回调：`编码中: {current}/{total}帧 | {fps} FPS | 剩余 {time}`

**进度条行为**：
- 推理阶段：0% → 100%
- 编码阶段：进度条重置，0% → 100%
- 状态标签显示当前阶段名

**单任务模式**：处理完成后自动清理 mask 缓存。Tab 1 的进度显示需适配新的三参数信号 `(current, total, phase)`，推理阶段和编码阶段分别显示。
**队列模式**：所有任务完成后清理缓存，或用户手动删除任务时清理对应缓存。

### 2.4 图片处理

图片模式不需要两阶段——直接推理并保存最终结果 PNG。续传支持通过跳过已存在的输出文件实现。

---

## 3. 数据模型

### 3.1 QueueTask

```python
@dataclass
class QueueTask:
    id: str                    # UUID
    input_path: str            # 输入文件/文件夹路径
    input_type: InputType      # VIDEO / IMAGE / IMAGE_FOLDER
    config: ProcessingConfig   # 独立的处理设置（模型、格式、背景模式）
    output_dir: str | None     # 输出目录（None = 与输入同目录）
    status: TaskStatus         # PENDING / PROCESSING / PAUSED / COMPLETED / FAILED / CANCELLED
    progress: int              # 当前帧/图片数（推理阶段）
    total: int                 # 总帧/图片数
    phase: ProcessingPhase     # INFERENCE / ENCODING / DONE
    error: str | None          # 错误信息
    created_at: float          # 创建时间戳

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingPhase(Enum):
    INFERENCE = "inference"
    ENCODING = "encoding"
    DONE = "done"
```

### 3.2 工程文件 (.brm)

JSON 格式，保存在 `~/.birefnet-gui/queue.brm`：

```json
{
  "version": 1,
  "updated_at": "2026-04-04T10:30:00",
  "tasks": [
    {
      "id": "a1b2c3d4",
      "input_path": "/Users/xxx/wedding.mp4",
      "input_type": "video",
      "config": {
        "model_name": "BiRefNet-general",
        "output_format": "mov_prores",
        "background_mode": "transparent"
      },
      "output_dir": null,
      "status": "processing",
      "progress": 80,
      "total": 100,
      "phase": "inference",
      "error": null,
      "created_at": 1712200000.0
    }
  ]
}
```

**自动保存时机**：
- 每 2 秒或每 100 帧（取先到者）
- 任务状态变更时（开始/暂停/完成/失败）
- 软件关闭时（`closeEvent` 中自动暂停并保存）

**启动时恢复**：
- 自动加载 `queue.brm`（无弹窗确认）
- 加载到队列 Tab
- 「开始队列」时跳过已完成的任务

---

## 4. 缓存管理

### 4.1 缓存目录结构

```
~/.birefnet-gui/cache/
├── {task_id}/
│   ├── masks/
│   │   ├── 000000.png    # grayscale alpha mask
│   │   ├── 000001.png
│   │   └── ...
│   └── metadata.json     # 原文件信息用于校验
└── ...
```

### 4.2 MaskCacheManager

```python
class MaskCacheManager:
    def __init__(self, cache_dir: str):
        ...

    def save_mask(self, task_id: str, frame_idx: int, mask: np.ndarray) -> None:
        """保存 alpha mask 为 grayscale PNG。"""

    def load_mask(self, task_id: str, frame_idx: int) -> np.ndarray:
        """读取缓存的 mask。"""

    def get_cached_count(self, task_id: str) -> int:
        """已缓存的帧数（用于确定续传起点）。"""

    def validate(self, task_id: str, video_info: dict) -> bool:
        """校验缓存是否与原视频匹配（路径、尺寸、帧率、总帧数）。"""

    def save_metadata(self, task_id: str, video_info: dict) -> None:
        """保存原视频元数据供校验。"""

    def cleanup(self, task_id: str) -> None:
        """删除指定任务的所有缓存。"""

    def cleanup_all(self) -> None:
        """删除所有缓存（用于清空队列时）。"""
```

### 4.3 缓存校验

`metadata.json` 内容：
```json
{
  "input_path": "/path/to/video.mp4",
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "frame_count": 100
}
```

续传时校验原文件是否一致。如果文件被移动/修改，提示用户重新处理。

---

## 5. 队列 Tab 界面

### 5.1 布局

```
┌───────────────────────────────────────────────────────────────┐
│  [单任务]  [批量队列 (3)]                                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  文件名              模型           格式        状态     │  │
│  │  wedding.mp4         general       ProRes     ✅ 完成   │  │
│  │  interview.mov       general       H.264      ▶ 推理 80/100 │
│  │  photos/             lite          PNG        ⏳ 等待    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  当前: interview.mov — 推理中 80/100帧 | 12 FPS | 剩余 0:10   │
│  ████████████████████░░░░░                                    │
│                                                               │
│  队列总进度: 任务 2/3 — 60%                                    │
│                                                               │
│  [开始队列]  [暂停]  [取消当前]  [清空队列]                     │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 交互

**任务列表** (QTableWidget)：
- 列：文件名、模型、格式、状态
- 支持行间拖拽排序
- 右键菜单：删除、移到顶部、移到底部
- 正在处理的任务不可删除/移动

**外部拖拽**：
- 拖入文件/文件夹到队列 Tab，自动添加任务
- 新任务继承 Tab 1 当前的默认设置

**控制按钮**：
- 「开始队列」：从第一个未完成任务开始顺序执行
- 「暂停」：暂停当前任务（保存进度）
- 「取消当前」：取消当前任务，开始下一个
- 「清空队列」：删除所有任务 + 清理缓存

**完成提示音**：
- 全部队列完成后播放系统提示音
- 使用 `QApplication.beep()` 或平台原生提示音

### 5.3 队列总进度

```
总进度 = (已完成任务帧数之和 × 2 + 当前任务进度) / (所有任务帧数之和 × 2)
```

其中当前任务进度 = 推理阶段完成帧数 + 编码阶段完成帧数（每阶段 0~total）。

×2 是因为每个任务有推理和编码两个阶段。

---

## 6. QueueManager

```python
class QueueManager(QObject):
    """队列逻辑管理器，不涉及 UI。"""

    # 信号
    task_added = pyqtSignal(str)          # task_id
    task_removed = pyqtSignal(str)        # task_id
    task_updated = pyqtSignal(str)        # task_id（状态/进度变更）
    queue_started = pyqtSignal()
    queue_paused = pyqtSignal()
    queue_finished = pyqtSignal()

    def add_task(self, task: QueueTask) -> None: ...
    def remove_task(self, task_id: str) -> None: ...
    def move_task(self, task_id: str, new_index: int) -> None: ...
    def clear_all(self) -> None: ...

    def start(self) -> None: ...
    def pause(self) -> None: ...
    def cancel_current(self) -> None: ...

    def save_to_file(self, path: str) -> None: ...
    def load_from_file(self, path: str) -> None: ...

    @property
    def tasks(self) -> list[QueueTask]: ...
    @property
    def is_running(self) -> bool: ...
```

**任务执行逻辑**：
1. `start()` 找到第一个 PENDING 或 PAUSED 的任务
2. 创建 MattingWorker（传入 `use_cache=True` + `start_frame`）
3. Worker 完成后，标记任务为 COMPLETED，开始下一个
4. Worker 出错后，标记 FAILED，继续下一个
5. 所有任务完成/失败后，发射 `queue_finished` 信号

---

## 7. Pipeline 修改

### 7.1 MattingPipeline 两阶段

```python
class MattingPipeline:
    def infer_phase(self, input_path, task_id, cache_manager,
                    start_frame=0, pause_event, cancel_event,
                    progress_callback) -> None:
        """推理阶段：从 start_frame 开始推理，保存 mask 到缓存。"""
        reader = FrameReader(input_path)
        model = load_model(...)
        for idx, frame in enumerate(reader):
            if idx < start_frame:
                continue
            # 暂停/取消检查
            mask = predict(model, frame)
            cache_manager.save_mask(task_id, idx, mask)
            progress_callback(idx + 1, total, "inference")

    def encode_phase(self, input_path, output_path, task_id,
                     cache_manager, config, pause_event, cancel_event,
                     progress_callback) -> None:
        """编码阶段：读原帧 + 缓存 mask → compose → 写入。"""
        reader = FrameReader(input_path)
        writer = create_writer(output_path, config, ..., audio_source=input_path)
        for idx, frame in enumerate(reader):
            mask = cache_manager.load_mask(task_id, idx)
            composed = compose_frame(frame, mask, config.background_mode)
            writer.write_frame(composed)
            progress_callback(idx + 1, total, "encoding")
```

### 7.2 MattingWorker 修改

```python
class MattingWorker(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, phase ("inference"/"encoding")

    def __init__(self, config, models_dir, input_path, output_path,
                 input_type=InputType.VIDEO, task_id=None, start_frame=0):
        ...
        self._task_id = task_id or str(uuid.uuid4())
        self._start_frame = start_frame
        self._cache = MaskCacheManager(CACHE_DIR)

    def run(self):
        if self._input_type == InputType.VIDEO:
            pipeline = MattingPipeline(self._config, self._models_dir)
            pipeline.infer_phase(self._input_path, self._task_id, self._cache,
                                 self._start_frame, ...)
            pipeline.encode_phase(self._input_path, self._output_path,
                                  self._task_id, self._cache, self._config, ...)
            self._cache.cleanup(self._task_id)  # 单任务模式清理
        else:
            # 图片模式不变
            ...
```

---

## 8. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/core/cache.py` | **新增** | MaskCacheManager：mask 保存/读取/清理/校验 |
| `src/core/queue_manager.py` | **新增** | QueueManager：队列逻辑、任务调度、.brm 持久化 |
| `src/core/queue_task.py` | **新增** | QueueTask 数据类、TaskStatus/ProcessingPhase 枚举 |
| `src/core/pipeline.py` | **修改** | MattingPipeline 拆分为 infer_phase + encode_phase |
| `src/core/image_pipeline.py` | **修改** | 续传支持（跳过已存在的输出文件） |
| `src/core/config.py` | **修改** | 新增 ProcessingPhase 枚举（如果不放 queue_task.py） |
| `src/gui/main_window.py` | **修改** | QTabWidget 包裹，新增"加入队列"按钮 |
| `src/gui/queue_tab.py` | **新增** | 队列 Tab 界面：任务列表、进度条、控制按钮 |
| `src/worker/matting_worker.py` | **修改** | 支持两阶段处理、task_id、start_frame |
| `src/worker/queue_worker.py` | **新增** | QueueWorker：逐任务调度 MattingWorker |

---

## 9. 测试计划

| 测试文件 | 测试点 |
|----------|--------|
| `tests/test_cache.py` | mask 保存/读取 round-trip；get_cached_count；validate 匹配/不匹配；cleanup |
| `tests/test_queue_manager.py` | 添加/删除/移动任务；save/load .brm 文件；清空队列 |
| `tests/test_queue_task.py` | QueueTask 序列化/反序列化 |
| `tests/test_pipeline_two_phase.py` | 两阶段 pipeline 端到端；断点续传（中断后从 start_frame 继续） |

---

## 10. 实现顺序

1. **数据模型** — QueueTask、TaskStatus、ProcessingPhase
2. **MaskCacheManager** — mask 缓存核心逻辑
3. **Pipeline 两阶段重构** — MattingPipeline.infer_phase + encode_phase
4. **MattingWorker 适配** — 两阶段 + 进度信号改造
5. **QueueManager** — 队列逻辑 + .brm 持久化
6. **QueueWorker** — 逐任务执行线程
7. **MainWindow Tab 改造** — QTabWidget + "加入队列"按钮
8. **Queue Tab 界面** — 任务列表 + 控制按钮 + 进度
9. **拖拽排序 + 右键菜单** — 交互增强
10. **完成提示音** — 队列完成后播放
