
# AI 数据分类分级系统 (AI-Powered Data Classification & Grading System)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![React: 18.x](https://img.shields.io/badge/react-18.x-61dafb)](https://reactjs.org/)

基于大语言模型（LLM）和深度学习技术实现的企业级数据资产处理系统，采用前后端分离架构，核心能力包括基于 Llama 的智能分类和基于 BERT 的敏感度分级。

---

## 🌟 核心功能

- **智能分类模块 (基于 Llama)**
  - 交互式分析：输入单个字段名称，实时获取业务大类与子类结果
  - 批量处理：上传 TXT/CSV 文件，自动化处理海量字段并导出结果
- **敏感分级模块 (基于 BERT)**
  - 批量分级：针对大规模敏感数据设计的 BERT 微调模型，划定数据安全级别（Level 1-4）
  - 现代化 UI：基于 React + Tailwind CSS 构建，简洁直观，模块化设计

---

## 🛠️ 技术栈

| 类别     | 技术栈                                                                 |
|----------|-----------------------------------------------------------------------|
| 前端     | React 18, Vite, Tailwind CSS, Lucide Icons                            |
| 后端     | FastAPI (Python), Uvicorn, PyTorch, Transformers                      |
| 模型     | 分类: Llama-Chat 系列模型 (Prompt Engineering + 两层分类逻辑)         |
|          | 分级: BERT (基于敏感度语料库微调)                                     |

---

## 📂 项目结构

```bash
LLAMa/
├── backend/                  # Python 后端服务
│   ├── main.py               # FastAPI 主程序 (包含 Llama 逻辑)
│   ├── bert_model.py         # BERT 预测逻辑
│   └── requirements.txt      # 后端依赖列表
├── frontend/                 # React 前端工程
│   ├── src/
│   │   ├── App.jsx           # 核心业务组件
│   │   └── index.css         # Tailwind 样式引入
│   ├── index.html            # 入口 HTML (含 Tailwind CDN)
│   └── package.json          # 前端依赖配置
└── model/                    # 本地模型存放目录 (建议)
```

---

## 🚀 环境搭建与运行

### 1. 后端环境 (Python)

**准备工作**：安装 Python 3.8+ 及 CUDA 环境（如需 GPU 加速）

```bash
# 进入项目根目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 安装核心依赖
pip install fastapi uvicorn python-multipart torch transformers

# 启动 FastAPI 服务
uvicorn main:app --host 0.0.0.0 --port 8000
```
**注意**：请在 `main.py` 中修改 `MODEL_PATH` 为本地模型路径。显存不足时可开启 Mock 模式进行 UI 测试。

### 2. 前端环境 (Node.js)

**准备工作**：安装 Node.js (推荐 v18+)

```bash
# 进入前端目录
cd ../frontend

# 安装前端依赖
npm install --legacy-peer-deps

# 启动开发服务器
npm run dev
```
**访问地址**：[http://localhost:5173](http://localhost:5173)

---

## 📝 使用指南

- **数据分类**：
  - 切换至“分类模块”
  - 使用“单字段输入”快速测试模型对关键词（如：财务报表、用户身份证）的理解
  - 使用“文件批量分析”上传文件，系统将按行读取并生成结果列表
- **数据分级**：
  - 切换至“分级模块”
  - 上传包含待评估字段的文件，点击“开始处理”获取 1-4 级安全定级
- **导出结果**：
  - 点击表格右上角的“导出 CSV”按钮，保存 AI 分析结果

---

## ⚠️ 疑难解答

- **样式不生效/页面全黑**：
  检查网络连接，确保 CDN 加载正常：
  ```html
  <script src="https://cdn.tailwindcss.com"></script>
  ```
- **后端连接失败**：
  检查 `App.jsx` 中的 `API_BASE` 是否指向正确的后端 IP 和端口（默认 `http://localhost:8000/api`）
- **模型加载报错**：
  确认 `transformers` 版本与模型匹配，并检查 GPU 显存是否足够

---

## ⚖️ 许可证

本项目遵循 [MIT License](LICENSE) 开源协议。
