# P6 体验优化：缓存管理 + 系统通知 + 模型下载优化

## 概述

在 P1-P5 功能完备的基础上，优化用户体验：
1. **缓存管理 GUI** — 新建「设置」Tab，查看和清理 mask 缓存
2. **系统通知** — 处理完成时发送系统通知（macOS 通知中心 / Windows toast）
3. **模型下载优化** — 自定义下载地址、真实下载进度、下载失败重试

---

## 1. 缓存管理 GUI

### 1.1 位置

新建第四个 Tab「设置」，Tab 顺序：单任务 | 批量队列 | 模型管理 | 设置

### 1.2 UI 设计

「设置」Tab 包含多个分组框。缓存管理部分：

```
┌─ 缓存管理 ──────────────────────────────┐
│                                           │
│  缓存目录: ~/.birefnet-gui/cache          │
│  占用空间: 2.3 GB (计算中...)              │
│                                           │
│  [ 清理全部缓存 ]   [ 打开目录 ]            │
│                                           │
│  ⚠ 清理缓存将删除所有断点续传进度           │
│                                           │
└───────────────────────────────────────────┘
```

### 1.3 功能

| 元素 | 说明 |
|------|------|
| 缓存目录 | 显示 `~/.birefnet-gui/cache` 路径，只读 |
| 占用空间 | 启动时 / Tab 切换到时异步计算目录大小，显示 "计算中..." 直到完成 |
| 清理全部缓存 | 点击弹出确认对话框，确认后调用 `MaskCacheManager.cleanup_all()`，刷新大小 |
| 打开目录 | 用系统文件管理器打开缓存目录（`QDesktopServices.openUrl`） |
| 警告文字 | 静态提示，告知清理后无法断点续传 |

### 1.4 计算缓存大小

- 新增 `MaskCacheManager.get_total_size()` 方法，遍历缓存目录计算总大小
- GUI 层用 `QThread` 调用，避免卡 UI
- 格式化显示：< 1MB 显示 "0 MB"，< 1GB 显示 "XX MB"，>= 1GB 显示 "X.X GB"

---

## 2. 系统通知

### 2.1 方案

使用 PyQt6 原生 `QSystemTrayIcon.showMessage()`，不增加额外依赖。

### 2.2 实现

- 在 `MainWindow.__init__` 中创建 `QSystemTrayIcon`
- 托盘图标使用应用图标，需要设为 `setVisible(True)` 才能发通知
- 如果 `QSystemTrayIcon.isSystemTrayAvailable()` 返回 False，跳过通知（保留现有 beep）

### 2.3 触发时机

| 场景 | 通知内容 |
|------|---------|
| 单任务完成 | 标题: "处理完成"，正文: 输出文件路径 |
| 单任务出错 | 标题: "处理出错"，正文: 错误信息 |
| 队列全部完成 | 标题: "队列完成"，正文: "N 个任务已完成" |

### 2.4 注意事项

- 通知不替代现有的弹窗（`QMessageBox`），两者共存
- macOS 上需要应用图标才能显示通知

---

## 3. 模型下载优化

### 3.1 自定义下载地址

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
- 选择"自定义"时显示一个 `QLineEdit` 输入框
- 下载源配置保存到 `~/.birefnet-gui/settings.json`（仅保存下载源，不做全面设置持久化）
- `ModelDownloader.download_model()` 接受 `endpoint` 参数

### 3.2 真实下载进度

**现状：** `snapshot_download` 内部使用 tqdm 显示进度，但 GUI 中用的是 indeterminate 进度条。

**方案：** 通过 `huggingface_hub` 的 `tqdm_class` 参数传入自定义 tqdm 类，捕获进度。

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
- 显示下载速度和剩余时间（从 tqdm 获取 `format_dict`）
- 进度文字格式：`"下载 model.safetensors: 45% | 1.2 GB/2.7 GB | 15.3 MB/s"`

**注意：** `snapshot_download` 会下载多个文件，每个文件有独立的 tqdm。进度显示以当前文件为单位。

### 3.3 下载失败重试

**现状：** 下载失败后弹出错误对话框，用户只能关掉重新点下载。

**方案：**
- 下载失败时，弹窗增加「重试」按钮（使用 `QMessageBox` 自定义按钮）
- 重试时利用 `resume_download=True`，从断点继续而不是重头开始
- 模型卡片上如果检测到下载了一半的目录（有部分文件但不完整），显示「继续下载」而非「下载」

**检测不完整下载：**
- 如果模型目录存在但缺少关键文件（如 `config.json` 或 `model.safetensors`），视为不完整
- `ModelDownloader.is_installed()` 已经只检查目录是否存在，增加 `is_partial()` 方法检查完整性

---

## 4. 文件变更汇总

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/gui/settings_tab.py` | 设置 Tab：缓存管理 + 下载设置 |
| `src/gui/notifier.py` | 系统通知封装 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/gui/main_window.py` | 新增设置 Tab + 初始化 Notifier + 完成/出错时发通知 |
| `src/gui/queue_tab.py` | 队列完成时调用 Notifier |
| `src/gui/model_tab.py` | 真实下载进度 + 重试按钮 + 读取下载源设置 |
| `src/core/model_downloader.py` | `endpoint` 参数 + `is_partial()` + tqdm 回调 |
| `src/core/cache.py` | `get_total_size()` 方法 |

---

## 5. 测试计划

### 5.1 缓存管理

| 测试 | 说明 |
|------|------|
| `test_get_total_size_empty` | 空缓存返回 0 |
| `test_get_total_size_with_data` | 有数据时返回正确字节数 |
| `test_cleanup_all_resets_size` | 清理后大小归零 |
| `test_format_size` | MB/GB 格式化显示 |

### 5.2 系统通知

| 测试 | 说明 |
|------|------|
| `test_notifier_no_crash_without_tray` | 系统托盘不可用时不崩溃 |
| `test_notify_message_content` | 通知消息格式正确 |

### 5.3 模型下载

| 测试 | 说明 |
|------|------|
| `test_is_partial_empty_dir` | 空目录视为不完整 |
| `test_is_partial_complete` | 完整目录视为已安装 |
| `test_download_with_custom_endpoint` | 自定义端点传递正确 |
| `test_settings_save_load` | 下载源设置保存和读取 |

---

## 6. 不做的事

- 不做全面设置持久化（只持久化下载源地址）
- 不做按任务清理缓存（只有全部清理）
- 不做通知开关（默认开启）
- 不做托盘常驻（仅用于发通知）
- 不做多文件并行下载
