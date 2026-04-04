@echo off
chcp 65001 >nul
echo ============================================
echo   BiRefNet GUI - 启用 NVIDIA GPU 加速
echo ============================================
echo.
echo 此脚本将下载 CUDA 版 PyTorch（约 2.5GB）
echo 需要 NVIDIA 显卡和已安装的 CUDA 驱动
echo.
set /p confirm=是否继续？(Y/N):
if /i not "%confirm%"=="Y" (
    echo 已取消。
    pause
    exit /b
)

echo.
echo 正在下载并安装 CUDA 版 PyTorch...
echo 这可能需要几分钟，请耐心等待...
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --target "%~dp0_internal" --upgrade --no-deps

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   安装完成！请重启 BiRefNet-GUI 使用 GPU 加速
    echo ============================================
) else (
    echo.
    echo 安装失败，请检查网络连接和 pip 是否可用。
    echo 你也可以手动执行：
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --target "%~dp0_internal" --upgrade --no-deps
)

echo.
pause
