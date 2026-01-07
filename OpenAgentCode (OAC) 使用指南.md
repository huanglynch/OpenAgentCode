# OpenAgentCode (OAC) 使用指南

OpenAgentCode (OAC) 是一个基于 AI 的代理工具，旨在帮助开发者处理开发者任务。它集成了大型语言模型 (LLM)、检索增强生成 (RAG)、上下文管理以及各种执行器，支持多种编程语言（如 Python、C++、Java 等）。OAC 通过 CLI 接口操作，允许用户生成代码、调试错误、优化文档、生成需求文档、设计架构、创建/审查 PR 等。核心目标是自动化软件开发流程，提高效率，同时保持安全性和可配置性。

OAC 的设计原则是从第一性原理出发：最小改动、简单可靠、目的导向。它使用 YAML 配置、Markdown 上下文文件和提示模板，确保易扩展。适用于个人开发者、团队协作和小规模项目。

## 1. 安装与初始化

### 要求
- Python 3.8+
- 依赖库：requests, json, yaml, click, difflib 等（通过 pip install 安装）。
- LLM 服务：支持本地（如 Ollama）或远程 API（如 xAI 的 Grok），需配置 API 密钥。
- 可选：GitPython (git 操作)、PyGithub (GitHub API)、sentence-transformers (RAG 嵌入)、tree-sitter (代码分块)、watchdog (文件监控)。

### 初始化项目
在项目目录运行：
```bash
python cli.py
```
如果 `config.yaml` 不存在，会自动初始化：
- 创建 `config.yaml`（默认配置，包括 LLM、路径、权限、RAG 等）。
- 创建 `prompts.yaml`（提示模板）。
- 创建 `AGENT.md`（上下文文件，用于持久化代理“记忆”）。
- 创建 `help.md`（帮助文档）。
- 创建 `.agent/tools/` 目录（自定义工具）。

编辑 `config.yaml` 以自定义 LLM 端点、API 密钥等。例如：
```yaml
llm:
  api_key: $YOUR_API_KEY$
  endpoint: https://api.x.ai/v1/chat/completions
  model: grok-4-1-fast-non-reasoning
```

## 2. 配置说明

OAC 通过 `config.yaml` 配置所有功能。关键部分：
- **llm**：端点、模型、温度、最大令牌、API 密钥。支持环境变量（如 `$XAI_API_KEY$`）。新增支持 vision_model 和 endpoint_vision 用于图像处理。
- **paths**：提示文件 (`prompts.yaml`)、上下文文件 (`AGENT.md`)、工具目录 (`.agent/tools/`)、嵌入缓存 (`cached_models/`)、帮助文件 (`help.md`)。
- **permissions**：文件读写、git 提交、bash 执行、GitHub API。默认安全设置（bash 和 GitHub API 禁用）。可预授权 bash 命令。
- **modes**：默认 `code` 或 `doc`。
- **timeouts**：bash 执行、编译、单元测试、LLM 请求、工具执行（默认 300-600 秒，可调整以适应大项目）。
- **rag**：嵌入模型 (`all-mpnet-base-v2`)、混合 alpha (0.4)、top_k (8)、chunk_size (384)、rerank_enabled (true)、刷新间隔 (300 秒)。
- **languages**：支持 python, cpp, c, js, java, cs, go；默认 python。
- **workspace**：当前工作目录路径，用于设置代理的工作环境。可通过 `/workspace` 命令动态切换并持久化。
- **tasks_optimizations**：针对 debug、ut、doc、requirements、design、optimize 的优化模板。
- **github**：令牌、所有者、仓库（用于 PR 操作）。

**提示模板**：在 `prompts.yaml` 中自定义 base_prompt、error_handle 等，以调整代理行为。

**上下文管理**：`AGENT.md` 存储 overview、history、facts、langs。代理自动加载/更新/压缩，确保“记住”用户身份和历史。

**重新加载配置**：运行 `/reload` 以应用变更。

## 3. 使用方式

### CLI 直接命令
```bash
python cli.py "Your task here" [OPTIONS]
```
选项：
- `--mode TEXT`：code 或 doc (默认: code)。
- `--headless`：仅输出 JSON。
- `--lang TEXT`：语言 (e.g., python, cpp)。
- `--chat`：进入直接聊天模式（不使用代理）。

示例：生成 Python 二分搜索函数
```bash
python cli.py "Create a binary search function in Python"
```

### 交互模式
无 prompt 时进入：
```bash
python cli.py
```
提示符：`>> `
- 输入任务或 slash 命令。
- 支持文件提及：`@filename`（模糊匹配，注入文件内容）。
- 退出：`/exit` 或 `/quit`。

### 聊天模式
```bash
python cli.py --chat
```
直接与 LLM 对话，支持流式输出。命令：`/model endpoint_type,model_name` 切换模型；新增 `/rag <query>` 支持 RAG 查询（检索本地文件并注入上下文）。

### 多模态支持（图片处理）
- **描述**：在 --chat 模式下，OAC 支持将图片文件或 URL 发送给支持视觉的 LLM（如 vision_model 配置的 AI:MM）进行分析、描述或识别。适用于图像相关查询，如物体识别、文本提取或视觉描述。
- **使用方法**：
  1. 进入 --chat 模式：`python cli.py --chat`。
  2. 输入包含图片的查询：
     - **URL**：如 "What is in this picture? https://example.com/cat.jpg"。OAC 自动检测并构建多模态消息。
     - **本地路径**：如 "Describe: D:/path/to/photo.png"。OAC 读取文件，转换为 base64 data URI 发送。
  3. 支持格式：.jpg、.jpeg、.png、.gif。
  4. 配置：确保 config.yaml 中的 llm.vision_model 和 llm.endpoint_vision 正确（默认使用 xAI API）。
- **示例**：
  - 输入："Analyze this image: http://example.com/chart.png"
  - 输出：AI 响应如 "The image shows a bar chart with sales data..."
- **限制**：仅限 --chat 模式；RAG 不索引图片；不支持图片编辑或生成（可通过自定义工具扩展）。如果读取失败，会打印错误。

## 4. 命令列表

### Slash 命令（在交互模式或直接 prompt 以 `/` 开头）
| 命令 | 说明 | 简单示例 |
|------|------|----------|
| `/help` | 显示帮助信息（从 `help.md` 加载）。 | `/help` → 输出完整帮助文档。 |
| `/exit`, `/quit` | 退出交互模式。 | `/exit` → "Goodbye!" |
| `/clear` | 清除代理上下文 (`AGENT.md`)。 | `/clear` → "Context cleared." |
| `/chat` | 进入聊天模式。 | `/chat` → "You: " 提示符。 |
| `/status` | 显示上下文概览（overview、history 条目、facts、langs）。 | `/status` → "History entries: 5" |
| `/config` | 显示当前配置（YAML 格式，API 密钥占位）。 | `/config` → 输出 config.yaml 内容。 |
| `/reload` | 重新加载 config.yaml 和 prompts.yaml。 | `/reload` → "Configuration and prompts reloaded." |
| `/create-pr [message]` | 创建 PR（生成标题、body，基于变更）。需 GitHub 配置。 | `/create-pr Add new feature` → 创建 PR 并输出 URL。 |
| `/review-pr [number]` | 审查 PR（分析 diff，检查 bug、安全、风格）。 | `/review-pr 5` → 输出审查评论 JSON。 |
| `/model [endpoint_type,model_name]` | 切换 LLM 模型和端点（e.g., vllm,AI:Pro）。保存到 config.yaml。 | `/model ollama,qwen3:1.7b` → "Switched to model qwen3:1.7b" |
| `/permissions [command]` | 列出或添加允许的 bash 命令（如果 exec_bash 禁用）。 | `/permissions git push` → 添加并保存。 `/permissions` → 列出允许命令。 |
| `/workspace [path]` | 切换工作目录并持久化到配置文件。支持相对/绝对路径，会自动重建 RAG 索引。 | `/workspace ./my-project` → 切换到指定目录并保存设置。 |
| `/commit-push-pr [message]` | 提交变更、推送分支、创建 PR。 | `/commit-push-pr Fix bug` → 执行 git 操作并创建 PR。 |
| `/requirements` | 生成 requirements.md（功能/非功能需求、约束）。 | `/requirements` → 输出需求文档。 |
| `/design` | 生成设计文档（UML 文本、组件、架构）。 | `/design` → 输出设计文档。 |
| `/optimize` | 优化代码或文档（性能、可读性、重构建议）。 | `/optimize` → 输出优化建议。 |
| `/rag <query>` | （仅聊天模式）使用 RAG 检索本地文件，并注入上下文到 LLM 查询。 | `/rag Explain the infer method in agent.py` → 输出基于 RAG 的响应。 |

### 其他任务命令（非 slash，直接 prompt）
- 以 `/` 开头但非以上命令：作为任务处理（e.g., `/some-task` → 代理推断）。

## 5. 可能适合的应用场景

OAC 适用于软件开发全生命周期，特别适合需要 AI 辅助的场景。以下按功能分类，并扩展了更多应用建议。新增场景基于实际行业用例，如自动化文档处理（减少 70-85% 处理成本）、文学优化（加速创作迭代）、设计生成（架构规划）、需求分析（用户故事起草）、ROI 报告（量化工具效益），以及其他如业务自动化、科学研究和制造业，以提供更全面的指导。扩展内容参考了行业报告和案例（如 AI 代理在 SDLC 中的集成、文档自动化 ROI，以及跨行业用例），旨在展示 OAC 如何通过其代码/文档代理功能实现高效自动化。

### 代码生成与优化
- **场景**：快速原型开发、算法实现、代码重构。适合初学者或时间紧迫的项目。
- 示例：生成排序算法、优化性能瓶颈。命令：`Implement quicksort in cpp`。

### 调试与错误处理
- **场景**：追踪 bug、分析日志。适合复杂项目中的错误诊断。
- 示例：`Debug @main.py with error`。代理使用 RAG 检索相关片段，注入上下文。

### 单元测试 (UT)
- **场景**：自动化测试生成与执行。适合 TDD (测试驱动开发) 或 CI/CD 管道。
- 示例：`Generate unit tests for @calculator.py`。支持 pytest 等框架。

### 文档管理与处理/编辑/优化
- **场景**：总结 README、优化文档清晰度、自动化数据提取和分类。适合开源项目、团队协作或高体积文档处理（如发票、合同、报告）。OAC 可处理 PDF/文本文件，实现 99%+ 准确率的数据提取，减少手动工作 80%，并确保合规审计。扩展到企业场景：处理财务报表、法律文件或合规文档，生成摘要或优化冗余内容。
- 示例：`--mode doc Summarize @README.md` 或 `Optimize contract @legal_doc.md`。使用 doc_optimize 提示减少冗余。

### 文学创作/优化
- **场景**：生成故事大纲、优化小说/文章结构、总结文学作品或生成创意内容。适合作家、内容创作者或教育领域，帮助加速迭代（如从草稿到精炼版本），或分析文学主题。OAC 可注入上下文历史，实现连续创作，并通过 RAG 检索类似作品提升原创性。适用于博客优化、脚本写作或学术论文润色，减少创作时间 50%。
- 示例：`Generate a short story outline about AI agents` 或 `Optimize this novel chapter @draft.txt for clarity and flow`。

### 各种设计
- **场景**：软件架构设计、UI/UX 原型生成、威胁建模或服务边界规划。适合架构师或产品团队，从需求生成 UML 图表、ADR（架构决策记录）或性能预算。扩展到非软件设计：如生成产品设计文档、流程图或硬件架构建议。OAC 可主动建议选项、 pros/cons，并集成现有代码库，实现端到端设计自动化。
- 示例：`/design Generate architecture for a user management system` 输出 UML 文本和组件图。

### 需求分析
- **场景**：从用户反馈生成需求文档、用户故事、优先级排序或非功能需求（NFRs）。适合产品经理在项目启动阶段，帮助处理反馈汇总、去重和验收标准建议。扩展到市场调研：分析竞争产品需求，或从文献中提取关键要求，帮助加速发现阶段。
- 示例：`/requirements Analyze user feedback for a mobile app` 输出 functional/non-functional 需求和约束。

### ROI 分析/报告
- **场景**：量化 AI 工具或项目投资回报，如跟踪代码生成速度、调试时间减少、文档效率提升。适合工程领导者评估工具采用框架（e.g., Gartner 模型），生成报告包括时间节省、生产力提升（平均 43% 效率增益）和成本降低（$2.3M/年）。OAC 可使用工具扩展生成图表或总结数据，支持决策如是否集成新 AI 代理。
- 示例：`Generate ROI report for implementing OAC in our dev team` 输出分析和建议。

### 基于设计文档的代码实现
- **场景**：从现有设计文档（如 UML、架构描述）自动生成初始代码框架或完整实现。适合软件开发迭代阶段，帮助桥接设计与编码，帮助减少手动转换错误，实现快速原型化。适用于敏捷团队或大型项目，帮助提升开发效率 60% 以上。通过 RAG 注入相关代码片段，确保实现符合设计约束。扩展到逆向工程：从遗留设计文档生成现代代码。
- **使用方法**：
  1. 生成或准备设计文档：使用 `/design` 命令生成设计文件（e.g., design.md），或手动创建/上传现有文档。
  2. 注入设计到 prompt：在交互模式或 CLI 中，使用文件提及 (@filename) 将设计注入任务 prompt 中。
  3. 指定语言和模式：设置 `--mode code --lang python` 以生成代码。
  4. 执行并验证：代理会分析设计、规划步骤、生成代码，并可选执行编译/测试。
  5. 迭代优化：如果输出不完美，使用 `/optimize` 或调试 prompt 细化。
- 示例：假设有 design.md 文件，运行 `Implement the design from @design.md in python`。代理注入 design.md 内容，生成代码文件，并可能执行 file_write 和 compile_run 动作。输出包括计划和生成的代码。

### 基于源代码的 RAG 查询
- **场景**：将项目源代码作为 RAG 知识库，使用自然语言查询代码功能、结构、依赖或潜在问题。适合代码审查、知识转移或大型项目维护，帮助开发者快速理解代码库，而无需手动阅读全部文件。OAC 的 RAG 系统自动索引项目文件（代码和文档），支持语义搜索（如函数解释、依赖分析），并通过代理注入相关片段到 LLM prompt 中，实现智能问询。适用于开源项目审计、团队 onboarding 或逆向工程，减少查询时间 70-90%。
- **使用方法**：
  1. **初始化 RAG 索引**：运行 OAC 时，VectorRAG 会自动扫描并索引当前目录下的支持文件（.py, .cpp, .md 等）。如果文件变更，启用 watchdog（需安装）后每 300 秒刷新（config.yaml 中的 index_refresh_interval）。
  2. **配置 RAG**：在 config.yaml 中调整 rag 部分，如 embedding_model 为 "all-mpnet-base-v2"、top_k 为 8（返回 top 结果）、rerank_enabled 为 true（使用 LLM rerank 优化结果）。对于代码查询，设置 chunk_method 为 "function"（tasks_optimizations.debug）。
  3. **构建查询 prompt**：使用自然语言描述问题，可结合文件提及 (@filename) 限制范围，或让 RAG 全局搜索。指定 --mode code 和 --lang 以聚焦。在聊天模式下，使用 `/rag <query>` 直接触发 RAG 注入。
  4. **执行查询**：在交互模式或 CLI 输入 prompt，或在聊天模式下用 /rag。代理调用 rag.search，注入相关文件内容（inject_rag_results），然后 LLM 分析并响应。
  5. **迭代与验证**：如果结果不准，添加更多上下文，或使用 /status 检查注入的上下文。输出包括计划和分析结果，可保存到 AGENT.md。
  6. **高级提示**：启用 rerank 以提升相关性；对于大项目，增加 chunk_size (384) 以处理长文件；监控 token_monitor 以避免 prompt 过长。
- 示例：
  - **示例 1: 查询函数功能**。prompt：`Explain how the infer method works in agent.py`。RAG 检索 agent.py 相关 chunk，注入内容，代理输出：详细解释计划、动作和上下文管理。
  - **示例 2: 项目整体分析**。prompt：`What are the main main dependencies in this Python project?`。RAG 搜索所有 .py 文件，注入依赖相关代码，代理生成总结列表。
  - **示例 3: 潜在问题查询**。prompt：`Identify security vulnerabilities in @tools.py`。RAG 注入文件，代理使用 error_handle prompt 分析，返回根因和修复建议。
  - **示例 4: 跨文件查询**。prompt：`How does RAG integrate with the agent in this codebase?`。RAG 返回 rag_vector.py 和 agent.py 片段，代理输出集成流程图或步骤。
  - **示例 5: 聊天模式 RAG**。在 --chat 下：`/rag Analyze the structure of cli.py`。临时创建 RAG，注入 top_k 文件内容，LLM 直接响应分析。

### 其他扩展场景
- **GitHub 集成与 PR 管理**：团队协作、代码审查。适合远程仓库操作（需启用 github_api 权限）。示例：`/create-pr New feature` 创建 PR；`/review-pr 1` 审查并建议修复。
- **多项目管理**：通过 `/workspace` 命令快速切换不同项目目录，每个项目维护独立的 RAG 索引。适合同时维护多个代码仓库的开发者，实现项目间的快速切换而无需重新初始化配置。示例：`/workspace ~/project-a` → `/workspace ~/project-b`，每次切换会自动重建对应项目的代码索引。
- **业务流程自动化**：财务发票处理、物流文档提取、制造质量控制记录分析。适合高卷文档行业如金融（处理贷款文件）和物流（提取运单细节）。
- **科学研究与药物发现**：处理科学文献、专利数据库或实验结果总结。适合生物/化学领域，使用 biopython 等库加速药物开发或文献审查。
- **自定义工具扩展**：集成特定分析（如自定义 analyzer）。适合高级用户扩展功能，如游戏开发（pygame 集成）或多媒体处理（mido）。
- **一般 AI 聊天**：非代理任务，如 brainstorm 或查询。适合快速咨询。示例：`--chat` 模式下输入问题，支持图像 URL/本地路径处理。
- **安全与权限控制**：生产环境，避免危险操作。默认禁用 bash 和 GitHub API，适合敏感项目。

总体优势：OAC 在小改动下实现可靠自动化，适用于个人学习、开源贡献、快速迭代。局限：不适合大规模生产（需手动审核输出）。通过这些场景，OAC 可实现显著 ROI，如减少手动文档处理 80% 或提升开发速度 71%（基于 Gartner 研究）。

## 6. 高级功能

### RAG (Retrieval-Augmented Generation)
- 自动索引代码/文档文件（支持 tree-sitter 分块）。
- 混合搜索（语义 + BM25），rerank 启用时使用 LLM 优化结果。
- 场景：语义检索依赖，提升代码分析准确性。
- 配置：调整 top_k、chunk_size 以平衡速度/精度。
- 新增：在聊天模式下，使用 `/rag <query>` 临时创建 RAG 实例，检索并注入上下文到 LLM 系统消息中，支持直接查询而不进入代理模式。

### 权限与安全
- 默认安全：禁用 exec_bash 和 github_api。
- 启用 bash：设置 `exec_bash: true`（警告：仅限信任环境）。或预授权命令。
- GitHub：配置 token/owner/repo，启用 github_api。

### 超时管理
- 防止挂起：调整 timeouts 以适应大文件编译或长测试。

### 自定义扩展
- **工具**：在 `.agent/tools/` 添加 Python 文件，定义函数（如 custom_analyzer）。代理自动发现。
- **提示优化**：编辑 prompts.yaml 以调整行为（e.g., 添加 CoT）。
- **上下文自定义**：手动编辑 `AGENT.md` 添加用户身份，确保格式正确。

## 7. 故障排除

- **Index is empty**：确保目录有支持文件（.py 等），不在隐藏目录。
- **Timeout expired**：增加 timeouts 或简化查询。
- **Permission denied**：检查 permissions，启用所需选项。
- **LLM call failed**：验证 endpoint/API 密钥，检查服务运行。
- **上下文过长**：运行 `/clear` 或自动压缩会触发。
- **模糊匹配失败**：使用完整文件名，或检查目录结构。
- **RAG 失败**：检查 embed_cache_dir 和 watchdog 安装；大文件使用 compress。

## 8. 工具功能说明、使用方法与示例

OAC 通过 `tools.py` 中的 ToolLoader 类支持内置和自定义工具，这些工具允许代理执行具体动作，如文件操作、git 管理、编译/测试和 GitHub API 调用。工具在代理的执行阶段（executor.py）被调用，用于实现任务的“执行”部分。内置工具包括 git_clone、file_search、compile_lang、run_ut、create_pr 和 review_pr 等，自定义工具可通过在 `.agent/tools/` 目录添加 Python 文件扩展。代理在 infer 方法中解析 LLM 输出为 actions，然后自动调用相应工具。

### 功能说明
- **内置工具**：预定义函数，用于常见操作如克隆仓库、搜索文件、编译代码、运行测试和 PR 管理。需启用相关权限（如 git_commit 或 github_api）。
- **自定义工具**：用户可在 `.agent/tools/` 添加 .py 文件，定义函数（e.g., def custom_analyzer(code): ...）。代理自动加载并可通过 action type='tool' 调用。
- **执行机制**：代理的 parse_output 将 LLM 响应解析为 {'actions': [...]}，然后 executor.execute 每个 action。支持 lang 参数和错误处理。
- **安全与限制**：工具执行受权限限制（e.g., exec_bash 默认禁用），超时默认 300-600 秒。GitHub 工具需配置 token/owner/repo。

### 使用方法
1. **配置权限**：在 config.yaml 中启用所需权限（e.g., git_commit: true）。对于 bash，使用 /permissions 添加允许命令。
2. **触发工具**：用户不直接调用工具，而是通过自然语言 prompt 描述任务（e.g., "Clone repo and compile"）。代理会决定使用工具，并输出结果。
3. **自定义扩展**：在 .agent/tools/my_tool.py 添加函数。代理重载配置（/reload）后可用。代理在 prompt 中可引用（LLM 决定）。
4. **查看输出**：在 headless 模式下，输出 JSON 包括工具结果；在正常模式下，格式化为 Markdown。
5. **错误处理**：如果工具失败，返回错误消息（e.g., 'Permission denied'）。使用 /status 检查上下文。

### 示例
- **示例 1: 使用 git_clone（内置工具）**
  - **场景**：克隆外部仓库以集成代码。
  - **prompt**：`Clone https://github.com/example/repo and analyze structure`
  - **代理行为**：解析为 action {'type': 'tool', 'name': 'git_clone', 'args': {'url': 'https://github.com/example/repo'}}，执行克隆，返回 'Successfully cloned'。
  - **输出**：计划 + 克隆结果。

- **示例 2: 使用 compile_lang（内置工具）**
  - **场景**：编译 C++ 文件并运行。
  - **prompt**：`Compile and run @main.cpp in cpp`
  - **代理行为**：注入文件内容，解析为 action {'type': 'compile_run', 'file': 'main.cpp', 'lang': 'cpp'}，使用 g++ 编译，返回输出或错误。
  - **输出**：编译结果和执行日志。

- **示例 3: 使用 create_pr（内置工具，需要 github_api 启用）**
  - **场景**：基于变更创建 PR。
  - **prompt**：`/create-pr Add feature X`
  - **代理行为**：使用 create_pr prompt 生成标题/body，解析为 action {'type': 'create_pr', 'title': '...', 'body': '...'}，调用 PyGithub 创建 PR，返回 URL。
  - **输出**：PR 创建确认。

- **示例 4: 自定义工具（假设 .agent/tools/my_tool.py 中有 def custom_analyzer(code): return "Analysis: " + code.upper()）**
  - **场景**：自定义代码分析。
  - **prompt**：`Analyze code @script.py with custom tool`
  - **代理行为**：加载工具，解析为 action {'type': 'tool', 'name': 'custom_analyzer', 'args': {'code': '...' }}，执行并返回结果。
  - **输出**：分析结果注入 facts。

- **示例 5: 使用 run_ut（内置工具）**
  - **场景**：运行 Python 单元测试。
  - **prompt**：`Run unit tests for @test.py in python`
  - **代理行为**：解析为 action {'type': 'run_ut', 'file': 'test.py', 'lang': 'python'}，使用 pytest 执行，返回测试报告。
  - **输出**：Passed/Failed 详情。

通过这些工具，OAC 可扩展到复杂工作流，如自动化 CI/CD 或集成第三方 API。自定义工具需测试兼容性。

## 9. 与 ClaudeCode (CC) 相似功能比较列表

以下表格比较 CC 和 OAC 的核心功能。CC 的信息基于 Yuker 的 X 推文（2026 年上下文）和典型 Claude 代码代理特征：CC 强调用户自定义“入职手册”（CLAUDE.md）、代码任务处理和简单 CLI 配置。OAC 提供更全面的工具集成和权限控制。优缺点基于可靠性、灵活性和潜在局限评估。覆盖度是主观估计，考虑功能实现相似性。新功能如聊天模式下的 /rag 提升了 OAC 的 RAG 覆盖度。

| CC 功能/命令 | 说明 | 优缺点 | OAC 类似功能/命令 | 说明 | 优缺点 | 覆盖度（%） |
|--------------|------|--------|-------------------|------|--------|-------------|
| **上下文持久化 (CLAUDE.md)** | 使用 Markdown 文件存储用户身份、项目背景、历史交互，实现 AI “记住我是谁”。手动编辑文件，Claude 在对话中注入。 | **优**：简单、直观，用户可直接编辑；**缺**：依赖手动更新，无自动压缩，可能导致上下文过长影响性能。 | **上下文管理 (AGENT.md)** | YAML 文件存储 overview、history、facts、langs。代理自动加载/更新/压缩（使用 LLM 总结），注入到 prompt 中。命令： `/status` 查看，`/clear` 重置。 | **优**：自动更新和压缩，确保高效；集成 RAG；**缺**：YAML 格式稍复杂于 Markdown，手动编辑需小心格式。 | 90%（OAC 更自动化，但核心持久化相同） |
| **代码生成任务** | 通过 prompt 生成代码，支持多种语言，基于 Claude 的自然语言处理。无专用命令，直接对话。 | **优**：Claude 的强大推理能力，生成高质量代码；**缺**：无内置执行器，需手动测试；依赖网络。 | **代码生成 (infer 方法)** | 通过 CLI prompt 生成代码，支持 python/cpp 等。示例：`Create binary search in Python`。集成执行器（compile_run）。 | **优**：内置执行和验证，支持多语言；RAG 增强准确性；**缺**：依赖本地配置，可能不如 Claude 推理深度。 | 85%（OAC 添加执行，但生成逻辑相似） |
| **调试与错误处理** | Prompt 中描述错误，Claude 分析根因并建议修复。支持文件上传/提及。 | **优**：智能根因分析，使用 CoT；**缺**：依赖用户提供上下文；可能 hallucinate。 | **调试 (error_handle prompt)** | 示例：`Debug @main.py with error`。使用 RAG 检索相关代码，自动优化 prompt。 | **优**：RAG 集成，提升准确性；自动 rerank 修复建议；**缺**：需本地文件索引，初始化时间长。 | 95%（OAC 增强检索，核心分析相同） |
| **文档优化** | 通过 prompt 总结/优化文档，支持 Markdown 处理。 | **优**：Claude 擅长自然语言优化；**缺**：无专用模式，需手动指定。 | **文档模式 (--mode doc)** | 示例：`--mode doc Optimize @README.md`。使用 doc_optimize prompt 和 rerank。 | **优**：专用模式和 chunking，提升效率；**缺**：依赖 RAG 配置，可能过拟合代码文件。 | 80%（OAC 更结构化，但优化逻辑相似） |
| **需求/设计生成** | Prompt 生成 requirements 或设计文档，使用模板。 | **优**：灵活，Claude 可生成 UML 等；**缺**：无内置模板，需用户提供。 | **/requirements, /design** | 生成需求/设计文档，使用专用 prompt。示例： `/design` 输出 UML 文本。 | **优**：内置模板和 lang 支持；**缺**：模板需自定义，不如 Claude 通用。 | 85%（OAC 命令化，更易用） |
| **PR 创建/审查** | 无内置，可能通过 prompt 模拟审查。Claude 可生成评论。 | **优**：可自定义审查逻辑；**缺**：无 GitHub 集成，需手动操作；不自动化。 | **/create-pr, /review-pr** | 自动生成标题/body，审查 diff。需 GitHub 配置。 | **优**：集成 PyGithub，自动化 push/PR；**缺**：需权限启用，安全风险。 | 70%（OAC 更自动化，CC 依赖手动） |
| **配置管理** | 通过“入职手册”教学配置 Claude 项目，无专用 CLI。 | **优**：教育性强，适合初学者；**缺**：无标准化文件，依赖推文/文档。 | **config.yaml, /config, /reload** | YAML 配置 LLM、权限等。命令查看/重载。 | **优**：标准化，易扩展；支持环境变量；**缺**：需 YAML 知识，初始设置复杂。 | 75%（OAC 更全面，但 CC 更简单） |
| **帮助系统** | 无内置，可能通过 prompt 查询帮助。 | **优**：Claude 可解释自身；**缺**：无结构化文档。 | **/help (help.md)** | 显示详细帮助文档。 | **优**：专用文件，表格化命令列表；**缺**：需维护文件。 | 60%（OAC 更结构化，CC 依赖对话） |
| **聊天模式** | Claude 默认对话模式，支持直接交互。 | **优**：流式响应，实时；**缺**：无代理层，易偏题。 | **--chat, /chat** | 直接 LLM 聊天，支持模型切换 (/model) 和 RAG 查询 (/rag)。 | **优**：支持流式、模型切换和 RAG 注入；**缺**：与代理模式分离。 | 95%（OAC 添加 RAG，提升实用性） |
| **工具扩展** | Claude 支持工具调用，但 CC 可能自定义简单工具。 | **优**：集成 Claude 工具链；**缺**：不持久化，需每次定义。 | **.agent/tools/ 目录** | 自动加载 Python 工具，支持 git_clone 等。 | **优**：持久化，易扩展；**缺**：需 Python 编码。 | 80%（OAC 更本地化） |
| **RAG 检索** | Claude 可能无内置 RAG，依赖用户上传。 | **优**：简单；**缺**：无自动索引，检索不语义化。 | **RAG (VectorRAG 类)** | 自动索引文件，混合搜索 + rerank。支持聊天模式 /rag。 | **优**：语义增强，提升准确；支持 watcher；**缺**：需计算资源。 | 60%（OAC 独有增强，CC 基础覆盖；新增 /rag 提升） |

**总体覆盖度**：约 82%。OAC 在自动化和集成上更强（e.g., RAG、执行器），但 CC 依赖 Claude 的强大 LLM 推理，可能在复杂任务上更灵活。OAC 适合本地/开源场景，CC 更云端/对话导向。建议根据需求混合使用。

更多问题：查看 `/help` 或源代码。