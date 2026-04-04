# P4 设计：FP16 加速 + 多模型支持 + 模型管理 + 时序修复

**日期**: 2026-04-05
**基于**: PROGRESS.md P3 完成后的待开发功能列表
**前置**: P3 高级参数 + 显存检测 + 布局重构已完成

---

## 1. 目标

在现有功能基础上实现四个模块：
1. **FP16 自动加速** — CUDA 设备自动开启半精度推理，提速 30-50%
2. **多模型支持** — 支持加载全部 6 种 BiRefNet 模型
3. **模型管理 Tab** — 模型下载/删除/状态管理界面 + hf-mirror 镜像优先
4. **异常帧时序修复** — 检测并修复 mask 质量塌陷的坏帧，消除闪现

---

## 2. FP16 自动加速

### 2.1 原理

FP16 半精度推理在 CUDA 设备上可将推理速度提升 30-50%，显存占用减半，且对图像分割任务几乎无质量损失（alpha mask 精度 ~0.001 完全够用）。

### 2.2 实现策略

**模型加载不变**：仍以 FP32 加载模型权重，确保兼容性。

**推理时自动 cast**：在 `predict()` / `predict_batch()` 内部根据设备类型决定是否使用混合精度。

```python
# inference.py 中的推理逻辑
if device.type == "cuda":
    with torch.autocast("cuda", dtype=torch.float16):
        result = model(input_tensor)
else:
    # MPS / CPU 保持 FP32（MPS 的 FP16 支持不稳定）
    result = model(input_tensor)
```

### 2.3 设备策略

| 设备 | 精度 | 原因 |
|------|------|------|
| CUDA | FP16 (autocast) | 完美支持，显著加速 |
| MPS | FP32 | Apple MPS 的 FP16 有已知兼容性问题 |
| CPU | FP32 | CPU FP16 无加速收益 |

### 2.4 GUI 变化

无。完全自动，用户无感。

---

## 3. 多模型支持

### 3.1 模型清单

| 模型 key | 目录名 | HuggingFace repo | 简介 | 适用场景 |
|----------|--------|-------------------|------|---------|
| general | birefnet-general | zhengpeng7/BiRefNet | 通用模型，效果均衡 | 大多数场景（默认） |
| lite | birefnet-lite | zhengpeng7/BiRefNet_lite | 轻量快速，精度略低 | 显存不足/追求速度 |
| HR | birefnet-hr | zhengpeng7/BiRefNet_HR | 高分辨率优化 | 4K 视频 |
| matting | birefnet-matting | zhengpeng7/BiRefNet-matting | 专注 matting，边缘细腻 | 人像/头发丝细节 |
| HR-matting | birefnet-hr-matting | zhengpeng7/BiRefNet_HR-matting | HR + matting 结合 | 高分辨率人像 |
| dynamic | birefnet-dynamic | zhengpeng7/BiRefNet_dynamic | 动态分辨率输入 | 不同分辨率混合输入 |

### 3.2 共享模型注册表

将 `download_models.py` 中的 `MODELS` 字典提取到 `src/core/config.py` 作为统一的模型注册表，包含 key、目录名、repo_id、简介、适用场景、预估大小。`download_models.py` 和模型管理 Tab 都从此处读取。

```python
@dataclass
class ModelInfo:
    key: str                # "general"
    dir_name: str           # "birefnet-general"
    repo_id: str            # "zhengpeng7/BiRefNet"
    display_name: str       # "BiRefNet-general"
    description: str        # "通用模型，效果均衡"
    use_case: str           # "大多数场景（默认推荐）"
    size_mb: int            # 424

MODEL_REGISTRY: dict[str, ModelInfo] = { ... }
```

### 3.3 inference.py 改动

- `load_model(model_name, device)` — 根据 `model_name` 从 `MODEL_REGISTRY` 查找目录名，定位 `models/{dir_name}` 路径加载
- 加载前检查目录是否存在，不存在则抛出明确的 `ModelNotFoundError`

### 3.4 SettingsPanel 联动

- 模型下拉框只显示已下载的模型
- 下拉框下方加一个"管理模型..."链接，点击跳转到模型管理 Tab

---

## 4. 模型管理 Tab

### 4.1 布局

```
┌──────────┬──────────┬──────────┐
│  单任务   │ 批量队列  │ 模型管理  │
├──────────┴──────────┴──────────┤
│                                │
│  ┌──────────────────────────┐  │
│  │ ✅ BiRefNet-general  424MB │  │
│  │ 通用模型，效果均衡           │  │
│  │ 适用：大多数场景（默认推荐）   │  │
│  │                    [删除]   │  │
│  ├──────────────────────────┤  │
│  │ BiRefNet-lite       210MB  │  │
│  │ 轻量快速，精度略低           │  │
│  │ 适用：显存不足/追求速度       │  │
│  │                    [下载]   │  │
│  ├──────────────────────────┤  │
│  │ BiRefNet-matting    424MB  │  │
│  │ 专注 matting，边缘细腻       │  │
│  │ 适用：人像/头发丝细节         │  │
│  │                    [下载]   │  │
│  ├──────────────────────────┤  │
│  │ ...其余 3 个模型...         │  │
│  └──────────────────────────┘  │
│                                │
│  ┌──────────────────────────┐  │
│  │ [████████░░░] 下载中 65%    │  │
│  │ BiRefNet-lite  145/210 MB   │  │
│  │               [取消]        │  │
│  └──────────────────────────┘  │
│                                │
│  下载源: hf-mirror.com ✓       │  │
│  模型目录: ./models/            │  │
│                                │
└────────────────────────────────┘
```

### 4.2 模型卡片

每个模型显示为一张卡片，包含：
- 模型名称 + 大小
- 一行简介
- 适用场景说明
- 状态按钮：已安装 → [删除]；未安装 → [下载]；下载中 → 进度

### 4.3 下载逻辑

**下载源**：
- 优先 `hf-mirror.com`（设置 `HF_ENDPOINT=https://hf-mirror.com`）
- 连接失败（超时 10 秒）自动回退 `huggingface.co`
- 底部状态栏显示当前使用的下载源

**下载执行**：
- QThread 中执行 `huggingface_hub.snapshot_download()`
- `snapshot_download` 自带断点续传
- 通过定时轮询模型目录大小估算下载进度（`huggingface_hub` 不提供精确的字节级回调）
- 支持取消（终止下载线程）

**同时只允许下载一个模型**，下载中其他模型的 [下载] 按钮置灰。

### 4.4 删除逻辑

- 点击 [删除] → 确认对话框 "确定删除 BiRefNet-lite？模型文件将被移除。"
- 如果被删除的模型正在单任务 Tab 中使用 → 自动切回 general（如果 general 也被删了则切到任意已安装模型）
- 如果是最后一个模型 → 阻止删除，提示"至少保留一个模型"

### 4.5 首次启动

- 启动时检测 `models/` 目录下有无任何模型
- 如果没有 → 自动切到模型管理 Tab，顶部显示横幅提示："请先下载至少一个模型才能开始处理"
- [开始处理] 按钮在无模型时禁用

### 4.6 联动

| 事件 | 联动行为 |
|------|---------|
| 模型下载完成 | 单任务 Tab 模型下拉框刷新，新模型可选 |
| 模型删除 | 下拉框移除该模型；如果当前选中则切到 general |
| 点击"管理模型..." | 切换到模型管理 Tab |
| 首次启动无模型 | 自动切到模型管理 Tab + 禁用开始按钮 |

---

## 5. 异常帧时序修复

### 5.1 问题

BiRefNet 逐帧独立推理，个别帧 mask 质量突然塌陷（本该透明的区域闪现出来），导致视频播放时出现闪烁。

### 5.2 算法：异常帧检测 + 邻帧替换

**输入**：推理阶段产出的所有 mask（已缓存在磁盘上）

**处理流程**：
1. 顺序读取每帧 mask（灰度图，0-255）
2. 计算当前帧与前一帧的像素级 L1 差异均值：`diff = mean(|mask[i] - mask[i-1]|) / 255.0`
3. 同样计算当前帧与后一帧的差异
4. 如果**同时**满足：`diff_prev > threshold` 且 `diff_next > threshold`，标记为异常帧
   - "同时"条件避免误判正常的场景切换（场景切换时前后帧都不同）
5. 异常帧的 mask 替换为前后帧加权平均：`mask[i] = 0.5 * mask[i-1] + 0.5 * mask[i+1]`
6. 首帧只和第 2 帧比，尾帧只和倒数第 2 帧比（单侧检测，阈值放宽 ×1.5）
7. 修复后的 mask 覆写回缓存文件

**阈值**：默认 `0.15`（即平均每像素差异超过 15% 视为异常）。硬编码为常量，不暴露给用户。

### 5.3 Pipeline 集成

```
推理阶段 → 时序修复阶段（新增）→ 编码阶段
```

在 `pipeline.py` 的 `infer_phase` 完成后、`encode_phase` 开始前，插入 `temporal_fix_phase`。

**新增 `src/core/temporal.py`**：

```python
def detect_and_fix_outliers(
    cache: MaskCacheManager,
    task_id: str,
    total_frames: int,
    threshold: float = 0.15,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """检测并修复异常帧，返回修复的帧数。"""
    ...
```

### 5.4 性能

时序修复阶段只涉及：
- 顺序读取灰度 PNG（几百 KB/帧）
- numpy 数组减法 + 均值计算
- 偶尔的加权平均 + PNG 覆写

**无 GPU 运算**。2340 帧视频预计 **3-5 秒**完成，相比推理阶段可忽略。

### 5.5 GUI 变化

- **SettingsPanel**：高级参数区新增"时序修复"开关（QCheckBox），默认开启
- **图片输入时**隐藏此开关（图片无时序概念）
- **进度显示**：三阶段显示
  - `推理中 1523/2340 | 8.5 FPS`
  - `时序修复中 1200/2340`（不显示 FPS，因为极快）
  - `编码中 800/2340 | 45 FPS`

### 5.6 断点续传兼容

时序修复阶段直接覆写缓存文件。如果修复阶段中断：
- 下次恢复时重新跑一遍时序修复（几秒钟，不值得做增量检测）
- 不影响已有的推理缓存和编码断点续传逻辑

---

## 6. ProcessingConfig 扩展

```python
@dataclass
class ProcessingConfig:
    # ... 现有字段 ...
    # 新增
    temporal_fix: bool = True   # 时序修复开关
```

### 6.1 .brm 序列化兼容

新增 `temporal_fix` 字段使用默认值 `True`，旧 .brm 文件加载时缺失该字段自动填充默认值。

---

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/core/config.py` | **修改** | 新增 ModelInfo + MODEL_REGISTRY；ProcessingConfig 增加 temporal_fix |
| `src/core/inference.py` | **修改** | load_model 支持多模型；predict/predict_batch 加 FP16 autocast |
| `src/core/temporal.py` | **新增** | 异常帧检测 + 邻帧替换算法 |
| `src/core/pipeline.py` | **修改** | 推理和编码之间插入 temporal_fix_phase |
| `src/core/queue_task.py` | **修改** | 序列化兼容 temporal_fix 字段 |
| `src/worker/matting_worker.py` | **修改** | 支持三阶段进度信号 |
| `src/gui/model_tab.py` | **新增** | 模型管理 Tab：卡片列表 + 下载/删除 + 进度 |
| `src/gui/settings_panel.py` | **修改** | 模型下拉框只显示已下载模型 + "管理模型..."链接 + 时序修复开关 |
| `src/gui/main_window.py` | **修改** | 新增模型管理 Tab + 首次启动检测 + 三阶段进度显示 |
| `download_models.py` | **修改** | 复用 config.py 的 MODEL_REGISTRY |

---

## 8. 测试计划

| 模块 | 测试内容 |
|------|---------|
| config.py | ModelInfo/MODEL_REGISTRY 完整性；temporal_fix 默认值；.brm 兼容 |
| inference.py | FP16 autocast 在 CUDA mock 下正确启用；多模型加载路径；ModelNotFoundError |
| temporal.py | 正常帧不被修改；异常帧被检测并替换；边界帧（首/尾）处理；全部正常帧零修改；连续异常帧处理 |
| pipeline.py | 三阶段端到端；temporal_fix=False 跳过修复阶段 |
| model_tab.py | 模型状态检测（已安装/未安装）；下载触发；删除逻辑；最后一个模型不可删 |
| queue_task.py | 旧 .brm 加载兼容 temporal_fix 字段 |

---

## 9. 实现顺序

1. **config.py** — ModelInfo + MODEL_REGISTRY + temporal_fix 字段
2. **inference.py** — FP16 autocast + 多模型加载
3. **temporal.py** — 异常帧检测修复算法
4. **pipeline.py** — 三阶段流水线
5. **queue_task.py** — .brm 序列化兼容
6. **model_tab.py** — 模型管理 Tab 界面 + 下载/删除逻辑
7. **settings_panel.py** — 模型下拉框联动 + 时序修复开关
8. **main_window.py** — 集成模型管理 Tab + 首次启动检测 + 三阶段进度
9. **download_models.py** — 复用 MODEL_REGISTRY
