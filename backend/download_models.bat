@echo off
chcp 65001 >nul
echo ========================================
echo CosyVoice 模型下载脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo 请选择下载方式：
echo.
echo [1] ModelScope（推荐，中国用户，下载速度快）
echo [2] HuggingFace（海外用户）
echo.
set /p choice="请输入选项 (1 或 2): "

if "%choice%"=="1" goto modelscope
if "%choice%"=="2" goto huggingface
echo [错误] 无效选项
pause
exit /b 1

:modelscope
echo.
echo ========================================
echo 使用 ModelScope 下载模型
echo ========================================
echo.

REM 检查是否安装了 modelscope
python -c "import modelscope" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 ModelScope...
    pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
    if errorlevel 1 (
        echo [错误] ModelScope 安装失败
        pause
        exit /b 1
    )
)

echo [信息] 开始下载模型...
echo [提示] 模型较大（约 2-4GB），下载可能需要较长时间
echo.

REM 使用 Python 脚本下载模型（更可靠）
echo 1 | python download_models.py
if errorlevel 1 (
    echo [错误] 模型下载失败，请检查网络连接
    pause
    exit /b 1
)
goto end

:huggingface
echo.
echo ========================================
echo 使用 HuggingFace 下载模型
echo ========================================
echo.

REM 检查是否安装了 huggingface_hub
python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 HuggingFace Hub...
    pip install huggingface_hub
    if errorlevel 1 (
        echo [错误] HuggingFace Hub 安装失败
        pause
        exit /b 1
    )
)

echo [信息] 开始下载模型...
echo [提示] 模型较大（约 2-4GB），下载可能需要较长时间
echo.

REM 使用 Python 脚本下载模型（更可靠）
echo 2 | python download_models.py
if errorlevel 1 (
    echo [错误] 模型下载失败，请检查网络连接
    pause
    exit /b 1
)
goto end

:end
echo.
echo ========================================
echo 下载完成！
echo ========================================
echo.
echo 下一步：
echo 1. 在 .env 文件中添加以下配置：
echo    COSYVOICE_MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B
echo    COSYVOICE_USE_SFT=false
echo    COSYVOICE_SPEAKER_ID=中文女
echo.
echo 2. 重启后端服务
echo.
pause

