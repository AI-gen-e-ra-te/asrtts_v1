# Resona - 语音对话系统

Resona 是一个基于 Ollama 和 CosyVoice 的智能语音对话系统，支持实时语音识别、大语言模型对话和高质量语音合成。

---

## 📋 目录

1. [环境要求](#1-环境要求)
2. [完整安装步骤（新手必读）](#2-完整安装步骤新手必读)
3. [详细配置说明](#3-详细配置说明)
4. [启动和使用](#4-启动和使用)
5. [常见问题](#5-常见问题)

---

## 1. 环境要求

### 必需软件
- ✅ **Windows 10 / 11**
- ✅ **Python 3.10** (必须是 3.10 版本)
- ✅ **Node.js LTS**（[下载地址](https://nodejs.org/)）
- ✅ **Git**（[下载地址](https://git-scm.com/)）

### 必需服务
- ✅ **Ollama**（[下载地址](https://ollama.com)）
  - 建议显存 ≥ 8GB（CPU 亦可运行，但较慢）

### 可选功能
- ⭐ **CosyVoice TTS**（高质量语音合成，推荐安装）
  - 建议显存 ≥ 4GB
  - 磁盘空间 ≥ 4GB

---

## 2. 完整安装步骤（新手必读）

### 步骤 1：安装 Ollama

1. 访问 https://ollama.com 下载并安装 Ollama
2. 打开命令提示符（CMD）或 PowerShell，输入以下命令验证：
   ```bash
   ollama -v
   ```

### 步骤 2：下载 Ollama 模型

在命令提示符中输入：
```bash
ollama pull qwen2.5:7b
```
> **说明：** 这会下载约 4-5GB 的模型文件。下载完成后，用 `ollama list` 验证是否成功。

### 步骤 3：下载项目代码

```bash
git clone 参考我们最新版！！！！！
cd Auralis
```

### 步骤 4：创建 Python 虚拟环境

> **⚠️ 注意：必须使用 Python 3.10**

在 `backend` 目录下执行：
```bash
cd backend
py -3.10 -m venv venv
.\venv\Scripts\activate
```
**验证：** 命令行前面显示 `(venv)` 表示激活成功。

### 步骤 5：安装 Python 依赖

确保在 `backend` 目录下且虚拟环境已激活：
```bash
pip install -r requirements.txt
```
> **说明：** 这会安装项目运行所需的所有依赖（包括 CosyVoice 的支持库），可能需要 10 分钟左右。

### 步骤 6：配置 CosyVoice TTS（推荐）

如果需要高质量语音合成，请执行以下两步：

#### 1. 下载 CosyVoice 代码库
双击运行 `setup_cosyvoice.bat`
*   此脚本会自动克隆 CosyVoice 仓库。
*   **注意：** 它不会重复安装依赖，因为步骤 5 已经安装了。

#### 2. 下载预训练模型
双击运行 `download_models.bat`
*   脚本会提示选择下载源：
    *   输入 `1`：**ModelScope**（国内推荐，速度快）
    *   输入 `2`：**HuggingFace**（海外推荐）
*   它会自动下载 2 个核心模型（Fun-CosyVoice3-0.5B 和 CosyVoice2-0.5B）。

### 步骤 7：配置环境变量

1. 在 `backend` 目录下，复制 `.env.example` 为 `.env`，或者新建 `.env` 文件。
2. 填入以下内容：

```env
# LLM 配置
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# CosyVoice 配置 (如已执行步骤 6)
COSYVOICE_MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B
COSYVOICE_USE_SFT=false
COSYVOICE_SPEAKER_ID=中文女
```

### 步骤 8：启动后端服务

在 `backend` 目录下（确保 `(venv)` 已激活）：
```bash
python -m app.main
```
看到 `Uvicorn running on http://0.0.0.0:8000` 表示成功。**请保持此窗口开启。**

### 步骤 9：启动前端界面

1. 打开新的命令行窗口。
2. 进入前端目录并安装依赖：
   ```bash
   cd ..\frontend
   npm install
   ```
3. 启动前端：
   ```bash
   npm run dev
   ```
4. 浏览器访问：http://localhost:3000

---

## 3. 详细配置说明

### LLM 模型配置
*   `LLM_MODEL` 必须与 `ollama list` 显示的名称完全一致。
*   如果使用不同模型（如 `llama3`），请同步修改 `.env`。

### CosyVoice 模型切换
默认使用的是 `Fun-CosyVoice3-0.5B`。如果你下载了其他模型，可以修改 `COSYVOICE_MODEL_DIR`。
*   最新版路径：`pretrained_models/Fun-CosyVoice3-0.5B`
*   V2 版本路径：`pretrained_models/CosyVoice2-0.5B`

---

## 4. 常见问题

**Q: `python` 或 `pip` 命令找不到？**
*   A: 请确保安装了 Python 3.10 并勾选了 "Add to PATH"。

**Q: CosyVoice 模型下载失败？**
*   A: 尝试运行 `download_models.bat` 并选择选项 1 (ModelScope)。如果依然失败，请检查网络连接。

**Q: 启动后端报错缺少模块？**
*   A: 确保你已经激活了虚拟环境 `(venv)` 并完整运行了步骤 5 的 `pip install`。

**Q: 显存不足 (CUDA Out of Memory)？**
*   A: CosyVoice 需要约 4GB 显存。如果显存不足，它可能会自动切换到 CPU 运行（速度较慢），或者尝试关闭其他占用显存的程序。

---

## 📝 脚本说明

*   `setup_cosyvoice.bat`: 仅克隆 CosyVoice 代码库。
*   `download_models.bat`: 下载预训练模型（仅保留了最常用的两个模型）。
