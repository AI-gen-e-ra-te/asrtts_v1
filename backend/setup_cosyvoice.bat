@echo off
chcp 65001 >nul
echo ========================================
echo CosyVoice TTS 安装脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/4] 检查 CosyVoice 仓库...
if not exist "CosyVoice" (
    echo [信息] 正在克隆 CosyVoice 仓库...
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    if errorlevel 1 (
        echo [错误] 克隆失败，请检查网络连接
        pause
        exit /b 1
    )
) else (
    echo [信息] CosyVoice 仓库已存在
)

echo.
echo [2/4] 更新子模块...
cd CosyVoice
git submodule update --init --recursive
if errorlevel 1 (
    echo [警告] 子模块更新可能失败，继续执行...
)
cd ..

echo.

echo [3/4] 跳过安装 CosyVoice 依赖...
REM pip install -r CosyVoice\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

echo.
echo [4/4] 跳过安装基础依赖...
REM pip install torch torchaudio modelscope huggingface_hub soundfile librosa numpy transformers -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 运行 download_models.bat 下载预训练模型
echo 2. 在 .env 文件中配置 CosyVoice 参数
echo 3. 重启后端服务
echo.
pause

