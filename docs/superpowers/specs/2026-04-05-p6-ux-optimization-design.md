# P6 体验优化：缓存管理 + 系统通知 + 模型下载优化 + 配置目录

## 概述

在 P1-P5 功能完备的基础上，优化用户体验：
1. **配置目录管理** — 支持切换数据存放位置（用户目录/应用目录/自定义）
2. **缓存管理 GUI** — 新建「设置」Tab，查看和清理 mask 缓存
3. **系统通知** — 处理完成时发送系统通知（macOS 通知中心 / Windows toast）
4. **模型下载优化** — 自定义下载地址、真实下载进度、下载失败重试

---

## 1. 配置目录管理

### 1.1 背景

当前数据路径硬编码为 `~/.birefnet-gui/`，包含：
- `cache/` — mask 缓存（断点续传）
- `queue.brm` — 队列任务持久化
- `settings.json` — P6 新增的设置文件

模型目录 `{app_root}/models/` 单独管理，不跟配置目录走。

### 1.2 三种模式

| 模式 | data_dir | 适用场景 |
|------|----------|---------|
| 用户目录（默认） | `~/.birefnet-gui/` | 多用户共用电脑，各自数据隔离 |
| 应用目录 | `{app_root}/data/` | 便携版用户，数据跟着应用走 |
| 自定义 | 用户指定的路径 | 想把缓存放在大容量硬盘等 |

### 1.3 启动时如何确定 data_dir

```
启动
  → 查 {app_root}/config.json
  → 有？读取 data_dir 字段
  → 没有？查 ~/.birefnet-gui/config.json
  → 有？读取 data_dir 字段
  → 没有？默认 data_dir = ~/.birefnet-gui/
```

应用目录优先，便于便携版覆盖默认行为。

`config.json` 格式（极简）：
```json
{
  "data_dir": "/path/to/data"
}
```

### 1.4 切换配置目录的逻辑

用户在设置 Tab 修改配置目录后：
1. 将 `config.json` 写到新的 data_dir 中
2. 如果选的是"应用目录"，额外在 `{app_root}/config.json` 写一份（确保下次启动能找到）
3. 弹窗提示"重启后生效"（不做热切换，避免复杂性）
4. 旧目录的数据不自动迁移，但提示用户可以手动复制

### 1.5 代码层面

新增 `src/core/data_dir.py`：
- `resolve_data_dir()` — 按上述优先级查找 config.json，返回 data_dir
- `get_cache_dir()` — `{data_dir}/cache/`
- `get_brm_path()` — `{data_dir}/queue.brm`
- `get_settings_path()` — `{data_dir}/settings.json`
- `save_config(data_dir, write_to_app_root=False)` — 写 config.json

替换现有硬编码：
- `src/worker/matting_worker.py` 的 `CACHE_DIR`
- `src/gui/main_window.py` 的 `BRM_PATH`

### 1.6 UI 设计

在「设置」Tab 中增加「数据目录」分组框：

```
┌─ 数据目录 ──────────────────────────────┐
│                                           │
│  数据存放位置: [用户目录 (默认)       ▼]  │
│                                           │
│  选项:                                     │
│    - 用户目录 (~/.birefnet-gui/)           │
│    - 应用目录 ({app_root}/data/)           │
│    - 自定义                                │
│                                           │
│  自定义路径: [__________________] [浏览]   │
│  (仅选择"自定义"时显示)                     │
│                                           │
│  当前生效: ~/.birefnet-gui/               │
│  ⚠ 修改后需重启生效                       │
│                                           │
└───────────────────────────────────────────┘
```

---

## 2. 缓存管理 GUI

### 2.1 位置

新建第四个 Tab「设置」，Tab 顺序：单任务 | 批量队列 | 模型管理 | 设置

### 2.2 UI 设计

```
┌─ 缓存管理 ──────────────────────────────┐
│                                           │
│  缓存目录: {data_dir}/cache               │
│  占用空间: 2.3 GB (计算中...)              │
│                                           │
│  [ 清理全部缓存 ]   [ 打开目录 ]            │
│                                           │
│  ⚠ 清理缓存将删除所有断点续传进度           │
│                                           │
└───────────────────────────────────────────┘
```

### 2.3 功能

| 元素 | 说明 |
|------|------|
| 缓存目录 | 显示实际缓存路径（基于当前 data_dir），只读 |
| 占用空间 | Tab 可见时异步计算目录大小，显示 "计算中..." 直到完成 |
| 清理全部缓存 | 点击弹出确认对话框，确认后调用 `MaskCacheManager.cleanup_all()`，刷新大小 |
| 打开目录 | 用系统文件管理器打开缓存目录（`QDesktopServices.openUrl`） |
| 警告文字 | 静态提示，告知清理后无法断点续传 |

### 2.4 计算缓存大小

- 新增 `MaskCacheManager.get_total_size()` 方法，遍历缓存目录计算总字节数
- GUI 层用 `QThread` 调用，避免卡 UI
- 格式化显示：< 1MB → "0 MB"，< 1GB → "XX MB"，>= 1GB → "X.X GB"

---

## 3. 系统通知

### 3.1 方案

使用 PyQt6 原生 `QSystemTrayIcon.showMessage()`，不增加额外依赖。

### 3.2 实现

- 在 `MainWindow.__init__` 中创建 `QSystemTrayIcon`
- 托盘图标使用应用图标，设为 `setVisible(True)` 才能发通知
- 如果 `QSystemTrayIcon.isSystemTrayAvailable()` 返回 False，跳过通知（保留现有 beep）

### 3.3 触发时机

| 场景 | 通知内容 |
|------|---------|
| 单任务完成 | 标题: "处理完成"，正文: 输出文件路径 |
| 单任务出错 | 标题: "处理出错"，正文: 错误信息 |
| 队列全部完成 | 标题: "队列完成"，正文: "N 个任务已完成" |

### 3.4 注意事项

- 通知不替代现有的弹窗（`QMessageBox`），两者共存
- macOS 上需要应用图标才能显示通知

---

## 4. 模型下载优化

### 4.1 自定义下载地址

**位置：**「设置」Tab 中新增「下载设置」分组框。

```
┌─ 下载设置 ──────────────────────────────┐
│                                           │
│  下载源: [hf-mirror.com (推荐)    ▼]      │
│                                           │
│  选项:                                     │
│    - hf-mirror.com (推荐)                  │
│    - huggingface.co (官方)                 │
│    - 自定义                                │
│                                           │
│  自定义地址: [________________________]     │
│  (仅选择"自定义"时显示)                     │
│                                           │
└───────────────────────────────────────────┘
```

**实现：**
- `QComboBox` 选择下载源：hf-mirror.com / huggingface.co / 自定义
- 选择"自定义"时显示 `QLineEdit` 输入框
- 下载源配置保存到 `{data_dir}/settings.json`
- `ModelDownloader.download_model()` 接受 `endpoint` 参数

### 4.2 真实下载进度

**现状：** `snapshot_download` 内部使用 tqdm 显示进度，但 GUI 用 indeterminate 进度条。

**方案：** 通过 `huggingface_hub` 的 `tqdm_class` 参数注入自定义 tqdm 类，捕获进度回调。

```python
from tqdm import tqdm

class QtProgressTqdm(tqdm):
    """Custom tqdm that emits Qt signals instead of printing."""
    def __init__(self, *args, signal=None, **kwargs):
        self._signal = signal
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self._signal and self.total:
            self._signal.emit(self.n, self.total, self.desc or "")
```

**UI 变化：**
- 模型 Tab 的进度条从 indeterminate 改为确定进度（百分比）
- 显示下载速度和剩余时间
- 进度文字格式：`"下载 model.safetensors: 45% | 1.2 GB/2.7 GB | 15.3 MB/s"`

**注意：** `snapshot_download` 会下载多个文件，每个文件有独立的 tqdm。进度显示以当前文件为单位。

### 4.3 下载失败重试

**现状：** 下载失败后弹出错误对话框，用户只能关掉重新点下载。

**方案：**
- 下载失败时，弹窗增加「重试」按钮（`QMessageBox` 自定义按钮）
- 重试利用 `resume_download=True`，从断点继续
- 模型卡片上如果检测到不完整下载，显示「继续下载」而非「下载」

**检测不完整下载：**
- 新增 `ModelDownloader.is_partial(model_key)` — 目录存在但缺少 `config.json`
- `ModelDownloader.is_installed()` 增强：目录存在 且 `config.json` 存在

---

## 5. 文件变更汇总

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/data_dir.py` | 配置目录解析：resolve_data_dir + 路径工具函数 |
| `src/gui/settings_tab.py` | 设置 Tab：数据目录 + 缓存管理 + 下载设置 |
| `src/gui/notifier.py` | 系统通知封装 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/gui/main_window.py` | 新增设置 Tab + 初始化 Notifier + 使用 data_dir 模块 |
| `src/gui/queue_tab.py` | 队列完成时调用 Notifier |
| `src/gui/model_tab.py` | 真实下载进度 + 重试按钮 + 读取下载源设置 |
| `src/core/model_downloader.py` | `endpoint` 参数 + `is_partial()` + tqdm 回调 |
| `src/core/cache.py` | `get_total_size()` 方法 |
| `src/worker/matting_worker.py` | `CACHE_DIR` 改用 `data_dir` 模块 |

---

## 6. 测试计划

### 6.1 配置目录

| 测试 | 说明 |
|------|------|
| `test_resolve_default` | 无 config.json 时返回默认路径 |
| `test_resolve_app_root_priority` | 应用目录 config.json 优先于用户目录 |
| `test_resolve_user_dir` | 只有用户目录有 config.json 时使用它 |
| `test_save_config` | 写入 config.json 内容正确 |
| `test_get_cache_dir` | 基于 data_dir 拼接正确 |
| `test_get_brm_path` | 基于 data_dir 拼接正确 |

### 6.2 缓存管理

| 测试 | 说明 |
|------|------|
| `test_get_total_size_empty` | 空缓存返回 0 |
| `test_get_total_size_with_data` | 有数据时返回正确字节数 |
| `test_cleanup_all_resets_size` | 清理后大小归零 |
| `test_format_size` | MB/GB 格式化显示 |

### 6.3 系统通知

| 测试 | 说明 |
|------|------|
| `test_notifier_no_crash_without_tray` | 系统托盘不可用时不崩溃 |
| `test_notify_message_content` | 通知消息格式正确 |

### 6.4 模型下载

| 测试 | 说明 |
|------|------|
| `test_is_partial_empty_dir` | 空目录视为不完整 |
| `test_is_partial_complete` | 完整目录视为已安装 |
| `test_download_with_custom_endpoint` | 自定义端点传递正确 |
| `test_settings_save_load` | 下载源设置保存和读取 |

---

## 7. 不做的事

- 不做全面设置持久化（只持久化下载源和配置目录）
- 不做按任务清理缓存（只有全部清理）
- 不做通知开关（默认开启）
- 不做托盘常驻（仅用于发通知）
- 不做多文件并行下载
- 不做配置目录热切换（修改后需重启）
- 不做旧数据自动迁移（提示用户手动复制）
- 模型目录不跟配置目录走（保持在应用目录）
