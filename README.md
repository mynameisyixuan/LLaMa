```markdown
# AI 数据分类分级系统 (AI-Powered Data Classification & Grading System)

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python: 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![React: 18.x](https://img.shields.io/badge/react-18.x-61dafb)

这是一个基于大语言模型（LLM）和深度学习技术实现的企业级数据资产处理系统。系统采用前后端分离架构，核心能力包括基于 Llama 的智能分类和基于 BERT 的敏感度分级。

## 🌟 核心功能

- **智能分类模块 (基于 Llama)**
  - 交互式分析：输入单个字段名称，实时获取业务大类与子类结果。
  - 批量处理：上传 TXT/CSV 文件，自动化处理海量字段并导出结果。
- **敏感分级模块 (基于 BERT)**
  - 批量分级：专门针对大规模敏感数据设计的 BERT 微调模型，划定数据安全级别（Level 1-4）。
  - 现代化 UI：基于 React + Tailwind CSS 构建，简洁直观，模块化设计。

## 🛠️ 技术栈

- **前端**: React 18, Vite, Tailwind CSS, Lucide Icons
- **后端**: FastAPI (Python), Uvicorn, PyTorch, Transformers
- **模型**:
  - 分类: Llama-Chat 系列模型 (采用 Prompt Engineering 与两层分类逻辑)
  - 分级: BERT (基于敏感度语料库微调)

## 📂 项目结构

```
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

## 🚀 环境搭建与运行

### 1. 后端环境 (Python)

**准备工作**：确保已安装 Python 3.8+ 及 CUDA 环境（如需 GPU 加速）。

```bash
# 进入项目根目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 安装核心依赖
pip install fastapi uvicorn python-multipart torch transformers
```

**运行后端**：

```bash
# 启动 FastAPI 服务
uvicorn main:app --host 0.0.0.0 --port 8000
```

**注意**：请在 `main.py` 中将 `MODEL_PATH` 修改为你本地模型的实际路径。如果显存不足，建议开启 Mock 模式进行 UI 测试。

### 2. 前端环境 (Node.js)

**准备工作**：确保已安装 Node.js (推荐 v18+)。

```bash
# 进入前端目录
cd ../frontend

# 安装前端依赖
npm install --legacy-peer-deps

# 启动开发服务器
npm run dev
```

**访问地址**：打开浏览器访问 [http://localhost:5173](http://localhost:5173)。

## 📝 使用指南

- **数据分类**：
  - 在顶部切换至“分类模块”。
  - 使用“单字段输入”快速测试模型对特定关键词（如：财务报表、用户身份证）的理解。
  - 使用“文件批量分析”上传文件，系统将按行读取并生成结果列表。
- **数据分级**：
  - 切换至“分级模块”。
  - 上传包含待评估字段的文件，点击“开始处理”获取 1-4 级安全定级。
- **导出结果**：
  - 点击表格右上角的“导出 CSV”按钮，即可将 AI 分析结果保存至本地。

## ⚠️ 疑难解答 (Troubleshooting)

- **样式不生效/页面全黑**：
  若本地 Tailwind 配置失败，本项目在 `index.html` 中通过 CDN 引入了样式脚本：
  ```html
  <script src="https://cdn.tailwindcss.com"></script>
  ```
  请确保网络连接正常以加载样式。
- **后端连接失败**：
  请检查 `App.jsx` 中的 `API_BASE` 变量是否正确指向了后端的 IP 和端口（默认 `http://localhost:8000/api`）。
- **模型加载报错**：
  检查 `transformers` 版本是否与模型匹配，并确认 GPU 显存是否足以支撑模型运行。

## ⚖️ 许可证

本项目遵循 [MIT License](LICENSE) 开源协议。
```
