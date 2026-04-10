# 主界面右侧参数持久化设计

## 1. 背景与目标

当前应用已具备：
- 队列状态持久化（`queue.brm`）
- 全局设置持久化（`settings.json`，当前主要用于下载源）

但主界面右侧 `SettingsPanel`（模型、格式、背景、编码等）在重启后会回到默认值，用户每次都要重新选择。

本设计目标：
- 仅持久化右侧参数（不包含输入路径、输出目录、当前 Tab 等界面状态）
- 复用现有 `settings.json`，避免新增文件
- 启动时自动加载上次配置
- 配置缺失/损坏/非法时自动回退默认，并在必要时自愈写回

## 2. 方案选择

### 2.1 备选方案

1) 扩展现有 `settings.json`（选中）
- 优点：改动最小、复用现有读写逻辑、与当前 data_dir 体系一致
- 缺点：`settings.json` 责任扩大（下载设置 + 面板默认值）

2) 新增 `panel_defaults.json`
- 优点：职责更清晰
- 缺点：新增文件与读写逻辑，维护成本更高

3) 复用 `config.json`
- 不选。`config.json` 当前用于 data_dir 定位，不应混入 UI 参数

### 2.2 结论

采用方案 1：在 `settings.json` 增加 `panel_defaults` 字段。

## 3. 数据模型设计

### 3.1 `AppSettings` 扩展

在 `src/core/settings.py` 的 `AppSettings` 增加字段：

- `panel_defaults: dict`

建议默认值为 `{}`（空字典），表示“未保存过面板参数，使用 UI 默认值”。

### 3.2 `panel_defaults` 字段定义

`panel_defaults` 存储以下键（与 `ProcessingConfig` 对齐）：

- `model_name` (str)
- `output_format` (str, enum value)
- `background_mode` (str, enum value)
- `bitrate_mode` (str, enum value)
- `custom_bitrate_mbps` (float)
- `encoding_preset` (str, enum value)
- `batch_size` (int)
- `inference_resolution` (int, enum value)
- `temporal_fix` (bool)
- `encoder_type` (str, enum value)

## 4. 读写流程设计

### 4.1 启动加载

时机：`MainWindow` 初始化 `SettingsPanel` 之后。

流程：
1. 调用 `load_settings(get_settings_path())`
2. 读取 `panel_defaults`
3. 若存在且有效：应用到 `SettingsPanel`
4. 若不存在：保持现有 UI 默认值
5. 若非法：忽略非法项并回退默认；必要时触发自愈写回

### 4.2 运行期保存

时机：`SettingsPanel.settings_changed` 信号触发时。

流程：
1. 从 `SettingsPanel.get_config()` 获取当前配置
2. 转为 `panel_defaults` 字典
3. 写回 `AppSettings.panel_defaults`
4. 调用 `save_settings(...)`

说明：初版不做节流，保持实现简单、状态一致性更高。后续如需优化可加 0.5~1s 防抖。

### 4.3 退出兜底

在 `MainWindow.closeEvent` 中，除现有队列保存外，再执行一次面板配置保存，确保异常顺序下仍可持久化最后状态。

## 5. 容错与自愈策略

### 5.1 文件级错误

- `settings.json` 不存在/JSON 解析失败/IO 失败：
  - `load_settings` 返回默认 `AppSettings`
  - UI 使用默认值
  - 下次保存时自动生成合法文件

### 5.2 字段级错误（`panel_defaults` 非法）

- 缺字段：该字段回退默认值
- 枚举值非法：该字段回退默认值
- 类型不匹配：该字段回退默认值

### 5.3 自愈写回触发条件

当检测到 `panel_defaults` 存在非法字段或非法值时：
- 使用“修复后的有效配置”覆盖 `panel_defaults`
- 调用 `save_settings` 写回，确保文件恢复为合法结构

## 6. 代码变更范围

- `src/core/settings.py`
  - 扩展 `AppSettings`，增加 `panel_defaults`
  - 更新 `to_dict` / `from_dict`
  - 增加对 `panel_defaults` 的基本类型校验

- `src/gui/settings_panel.py`
  - 新增 `apply_config(config: ProcessingConfig)`（或同等接口）
  - 按配置回填各控件选项

- `src/gui/main_window.py`
  - 启动后加载并应用 `panel_defaults`
  - 监听 `settings_changed` 并写回
  - `closeEvent` 增加面板配置兜底保存

- `tests/test_settings.py`
  - 新增 `panel_defaults` 读写、缺失回退、非法值回退/自愈测试

## 7. 验收标准

1. 首次启动（无 `settings.json`）
- 程序正常启动，右侧参数为默认值
- 用户修改参数后关闭程序，自动生成 `settings.json`

2. 二次启动
- 右侧参数恢复为上次值

3. 损坏配置
- 手动写入非法 `panel_defaults` 后启动不崩溃
- 对应字段回退默认
- 文件被修复为合法配置

4. 兼容旧版本
- 旧 `settings.json`（无 `panel_defaults`）可正常读取
- 不影响下载源等已有设置

## 8. 非目标（本次不做）

- 不持久化输入路径、输出目录、当前 Tab、窗口位置
- 不引入新文件（如 `panel_defaults.json`）
- 不实现复杂迁移器（按字段容错即可）

## 9. 风险与缓解

- 风险：启动阶段模型列表可能与已保存 `model_name` 不一致（模型被删除）
  - 缓解：`apply_config` 时若模型不存在，回退到当前可选第一个模型

- 风险：控件联动（格式变化导致可选项变化）影响回填顺序
  - 缓解：`apply_config` 内部按“格式 -> 模式/编码器 -> 高级参数”顺序设置，必要时暂时阻断信号

