# OpenAgentCode
An AI agent for developer tasks, integrating LLM, RAG, and tools.
Supports Ollama, vLLM for local/intranet, and cloud APIs.

# OpenAgentCode (OAC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

OpenAgentCode (OAC) is an AI-powered agent tool designed to assist developers with tasks like code generation, debugging, documentation optimization, and more. It integrates Large Language Models (LLM), Retrieval-Augmented Generation (RAG), context management, and various executors. Supports languages like Python, C++, Java, etc.

## Features
- **Code Tasks**: Generate, debug, optimize code.
- **Doc Tasks**: Summarize, optimize documents.
- **RAG Integration**: Semantic search over your codebase.
- **CLI Interface**: Easy-to-use commands.
- **GitHub Integration**: Create/review PRs (with permissions).
- **Multimodal**: Image analysis in chat mode.

## Installation
1. Clone the repo:
   ```
   git clone https://github.com/huanglynch/OpenAgentCode.git
   cd OpenAgentCode
   ```
2. Install dependencies:
   ```
   pip install requests pyyaml click difflib sentence-transformers
   ```
   - Optional: `pip install gitpython pygithub tree-sitter watchdog`.
3. Configure `config.yaml` (e.g., set LLM API key).
4. Run: `python cli.py`.

## Usage
- Interactive: `python cli.py`
- Direct: `python cli.py "Create binary search in Python"`
- Chat: `python cli.py --chat`

For full guide, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## License
MIT License. See [LICENSE](LICENSE) (create if needed).

<details>
<summary>Switch to Chinese / 切换到中文</summary>

# OpenAgentCode (OAC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

OpenAgentCode (OAC) 是一个基于 AI 的代理工具，旨在帮助开发者处理开发任务。它集成了大型语言模型 (LLM)、检索增强生成 (RAG)、上下文管理以及各种执行器，支持多种编程语言（如 Python、C++、Java 等）。

## 功能
- **代码任务**：生成、调试、优化代码。
- **文档任务**：总结、优化文档。
- **RAG 集成**：对代码库进行语义搜索。
- **CLI 接口**：易用命令。
- **GitHub 集成**：创建/审查 PR（需权限）。
- **多模态**：聊天模式下的图像分析。

## 安装
1. 克隆仓库：
   ```
   git clone https://github.com/huanglynch/OpenAgentCode.git
   cd OpenAgentCode
   ```
2. 安装依赖：
   ```
   pip install requests pyyaml click difflib sentence-transformers
   ```
   - 可选：`pip install gitpython pygithub tree-sitter watchdog`。
3. 配置 `config.yaml`（例如设置 LLM API 密钥）。
4. 运行：`python cli.py`。

## 使用
- 交互模式：`python cli.py`
- 直接命令：`python cli.py "Create binary search in Python"`
- 聊天模式：`python cli.py --chat`

完整指南见 [USAGE_GUIDE.md](USAGE_GUIDE.md)。

## 许可证
MIT 许可证。见 [LICENSE](LICENSE)。

</details>

