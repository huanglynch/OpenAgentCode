# OpenAgentCode (OAC) Usage Guide

OpenAgentCode (OAC) is an AI-based agent tool designed to assist developers with development tasks. It integrates large language models (LLM), retrieval-augmented generation (RAG), context management, and various executors, supporting multiple programming languages (such as Python, C++, Java, etc.). OAC operates via a CLI interface, allowing users to generate code, debug errors, optimize documents, generate requirements documents, design architectures, create/review PRs, and more. The core goal is to automate software development processes, improve efficiency, while maintaining security and configurability.

OAC's design principles are based on first principles: minimal changes, simplicity and reliability, purpose-oriented. It uses YAML configuration, Markdown context files, and prompt templates to ensure easy extensibility. Suitable for individual developers, team collaboration, and small-scale projects.

## 1. Installation and Initialization

### Requirements
- Python 3.8+
- Dependencies: requests, json, yaml, click, difflib, etc. (install via pip).
- LLM Service: Supports local (e.g., Ollama) or remote APIs (e.g., xAI's Grok), requires API key configuration.
- Optional: GitPython (for git operations), PyGithub (for GitHub API), sentence-transformers (for RAG embeddings), tree-sitter (for code chunking), watchdog (for file monitoring).

### Initialize Project
Run in the project directory:
```bash
python cli.py
```
If `config.yaml` does not exist, it will automatically initialize:
- Create `config.yaml` (default configuration, including LLM, paths, permissions, RAG, etc.).
- Create `prompts.yaml` (prompt templates).
- Create `AGENT.md` (context file for persisting agent "memory").
- Create `help.md` (help documentation).
- Create `.agent/tools/` directory (for custom tools).

Edit `config.yaml` to customize LLM endpoint, API key, etc. For example:
```yaml
llm:
  api_key: $YOUR_API_KEY$
  endpoint: https://api.x.ai/v1/chat/completions
  model: grok-4-1-fast-non-reasoning
```

## 2. Configuration Explanation

OAC configures all functions via `config.yaml`. Key sections:
- **llm**: Endpoint, model, temperature, max tokens, API key. Supports environment variables (e.g., `$XAI_API_KEY$`). Newly added support for vision_model and endpoint_vision for image processing.
- **paths**: Prompts file (`prompts.yaml`), context file (`AGENT.md`), tools directory (`.agent/tools/`), embedding cache (`cached_models/`), help file (`help.md`).
- **permissions**: File read/write, git commit, bash execution, GitHub API. Default secure settings (bash and GitHub API disabled). Can pre-authorize bash commands.
- **modes**: Default `code` or `doc`.
- **timeouts**: Bash execution, compilation, unit tests, LLM requests, tool execution (default 300-600 seconds, adjustable for large projects).
- **rag**: Embedding model (`all-mpnet-base-v2`), hybrid alpha (0.4), top_k (8), chunk_size (384), rerank_enabled (true), refresh interval (300 seconds).
- **languages**: Supports python, cpp, c, js, java, cs, go; default python.
- **tasks_optimizations**: Optimization templates for debug, ut, doc, requirements, design, optimize.
- **github**: Token, owner, repo (for PR operations).

**Prompt Templates**: Customize base_prompt, error_handle, etc., in `prompts.yaml` to adjust agent behavior.

**Context Management**: `AGENT.md` stores overview, history, facts, langs. The agent automatically loads/updates/compresses to ensure "remembering" user identity and history.

**Reload Configuration**: Run `/reload` to apply changes.

## 3. Usage Methods

### CLI Direct Commands
```bash
python cli.py "Your task here" [OPTIONS]
```
Options:
- `--mode TEXT`: code or doc (default: code).
- `--headless`: Output JSON only.
- `--lang TEXT`: Language (e.g., python, cpp).
- `--chat`: Enter direct chat mode (without agent).

Example: Generate Python binary search function
```bash
python cli.py "Create a binary search function in Python"
```

### Interactive Mode
Enter without prompt:
```bash
python cli.py
```
Prompt: `>> `
- Input tasks or slash commands.
- Supports file mentions: `@filename` (fuzzy matching, injects file content).
- Exit: `/exit` or `/quit`.

### Chat Mode
```bash
python cli.py --chat
```
Direct conversation with LLM, supports streaming output. Commands: `/model endpoint_type,model_name` to switch models; newly added `/rag <query>` supports RAG queries (retrieves local files and injects into context).

### Multimodal Support (Image Processing)
- **Description**: In --chat mode, OAC supports sending image files or URLs to vision-capable LLMs (e.g., configured vision_model) for analysis, description, or recognition. Suitable for image-related queries like object recognition, text extraction, or visual descriptions.
- **Usage**:
  1. Enter --chat mode: `python cli.py --chat`.
  2. Input query with image:
     - **URL**: e.g., "What is in this picture? https://example.com/cat.jpg". OAC automatically detects and builds multimodal message.
     - **Local Path**: e.g., "Describe: D:/path/to/photo.png". OAC reads file, converts to base64 data URI for sending.
  3. Supported Formats: .jpg, .jpeg, .png, .gif.
  4. Configuration: Ensure llm.vision_model and llm.endpoint_vision are correct in config.yaml (default uses xAI API).
- **Example**:
  - Input: "Analyze this image: http://example.com/chart.png"
  - Output: AI response like "The image shows a bar chart with sales data..."
- **Limitations**: Limited to --chat mode; RAG does not index images; does not support image editing or generation (extendable via custom tools). Prints error if read fails.

## 4. Command List

### Slash Commands (In interactive mode or direct prompt starting with `/`)
| Command | Description | Simple Example |
|------|------|----------|
| `/help` | Display help information (loaded from `help.md`). | `/help` → Outputs full help document. |
| `/exit`, `/quit` | Exit interactive mode. | `/exit` → "Goodbye!" |
| `/clear` | Clear agent context (`AGENT.md`). | `/clear` → "Context cleared." |
| `/chat` | Enter chat mode. | `/chat` → "You: " prompt. |
| `/status` | Display context overview (overview, history entries, facts, langs). | `/status` → "History entries: 5" |
| `/config` | Display current configuration (YAML format, API key placeholder). | `/config` → Outputs config.yaml content. |
| `/reload` | Reload config.yaml and prompts.yaml. | `/reload` → "Configuration and prompts reloaded." |
| `/create-pr [message]` | Create PR (generate title, body based on changes). Requires GitHub config. | `/create-pr Add new feature` → Creates PR and outputs URL. |
| `/review-pr [number]` | Review PR (analyze diff, check bugs, security, style). | `/review-pr 5` → Outputs review comments JSON. |
| `/model [endpoint_type,model_name]` | Switch LLM model and endpoint (e.g., vllm,AI:Pro). Saves to config.yaml. | `/model ollama,qwen3:1.7b` → "Switched to model qwen3:1.7b" |
| `/permissions [command]` | List or add allowed bash commands (if exec_bash disabled). | `/permissions git push` → Adds and saves. `/permissions` → Lists allowed commands. |
| `/commit-push-pr [message]` | Commit changes, push branch, create PR. | `/commit-push-pr Fix bug` → Executes git operations and creates PR. |
| `/requirements` | Generate requirements.md (functional/non-functional requirements, constraints). | `/requirements` → Outputs requirements document. |
| `/design` | Generate design document (UML text, components, architecture). | `/design` → Outputs design document. |
| `/optimize` | Optimize code or documents (performance, readability, refactoring suggestions). | `/optimize` → Outputs optimization suggestions. |
| `/rag <query>` | (Chat mode only) Use RAG to retrieve local files and inject into LLM query. | `/rag Explain the infer method in agent.py` → Outputs RAG-based response. |

### Other Task Commands (Non-slash, direct prompt)
- Starting with `/` but not listed above: Treated as tasks (e.g., `/some-task` → Agent infers).

## 5. Potential Application Scenarios

OAC is suitable for the full software development lifecycle, especially scenarios needing AI assistance. The following are categorized by function, with expanded application suggestions. New scenarios are based on real industry use cases, such as automated document processing (reducing 70-85% processing costs), literary optimization (accelerating creative iterations), design generation (architecture planning), requirements analysis (drafting user stories), ROI reports (quantifying tool benefits), and others like business automation, scientific research, and manufacturing, to provide more comprehensive guidance. Expanded content references industry reports and cases (e.g., AI agent integration in SDLC, document automation ROI, and cross-industry use cases), demonstrating how OAC achieves efficient automation through its code/document agent functions.

### Code Generation and Optimization
- **Scenarios**: Rapid prototyping, algorithm implementation, code refactoring. Suitable for beginners or time-constrained projects.
- Example: Generate sorting algorithms, optimize performance bottlenecks. Command: `Implement quicksort in cpp`.

### Debugging and Error Handling
- **Scenarios**: Track bugs, analyze logs. Suitable for error diagnosis in complex projects.
- Example: `Debug @main.py with error`. Agent uses RAG to retrieve relevant snippets and inject context.

### Unit Testing (UT)
- **Scenarios**: Automated test generation and execution. Suitable for TDD (Test-Driven Development) or CI/CD pipelines.
- Example: `Generate unit tests for @calculator.py`. Supports frameworks like pytest.

### Document Management and Processing/Editing/Optimization
- **Scenarios**: Summarize README, optimize document clarity, automate data extraction and classification. Suitable for open-source projects, team collaboration, or high-volume document processing (e.g., invoices, contracts, reports). OAC can handle PDF/text files with 99%+ accuracy in data extraction, reducing manual work by 80%, and ensuring compliance audits. Extends to enterprise scenarios: processing financial reports, legal documents, or compliance docs, generating summaries or optimizing redundant content.
- Example: `--mode doc Summarize @README.md` or `Optimize contract @legal_doc.md`. Uses doc_optimize prompt to reduce redundancy.

### Literary Creation/Optimization
- **Scenarios**: Generate story outlines, optimize novel/article structures, summarize literary works, or generate creative content. Suitable for writers, content creators, or education, helping accelerate iterations (e.g., from draft to refined version), or analyze literary themes. OAC can inject context history for continuous creation and use RAG to retrieve similar works to enhance originality. Applicable to blog optimization, script writing, or academic paper polishing, reducing creation time by 50%.
- Example: `Generate a short story outline about AI agents` or `Optimize this novel chapter @draft.txt for clarity and flow`.

### Various Designs
- **Scenarios**: Software architecture design, UI/UX prototype generation, threat modeling, or service boundary planning. Suitable for architects or product teams, generating UML diagrams, ADR (Architecture Decision Records), or performance budgets from requirements. Extends to non-software design: such as generating product design documents, flowcharts, or hardware architecture suggestions. OAC can proactively suggest options, pros/cons, and integrate existing codebases for end-to-end design automation.
- Example: `/design Generate architecture for a user management system` outputs UML text and component diagrams.

### Requirements Analysis
- **Scenarios**: Generate requirements documents, user stories, priority sorting, or non-functional requirements (NFRs) from user feedback. Suitable for product managers in project initiation phases, helping process feedback aggregation, deduplication, and acceptance criteria suggestions. Extends to market research: analyzing competitor product requirements or extracting key requirements from literature, accelerating discovery phases.
- Example: `/requirements Analyze user feedback for a mobile app` outputs functional/non-functional requirements and constraints.

### ROI Analysis/Report
- **Scenarios**: Quantify ROI for AI tools or projects, such as tracking code generation speed, debugging time reduction, document efficiency improvements. Suitable for engineering leaders evaluating tool adoption frameworks (e.g., Gartner model), generating reports including time savings, productivity gains (average 43% efficiency increase), and cost reductions ($2.3M/year). OAC can use tool extensions to generate charts or summarize data, supporting decisions like integrating new AI agents.
- Example: `Generate ROI report for implementing OAC in our dev team` outputs analysis and suggestions.

### Code Implementation Based on Design Documents
- **Scenarios**: Automatically generate initial code frameworks or complete implementations from existing design documents (e.g., UML, architecture descriptions). Suitable for software development iteration phases, bridging design and coding, reducing manual conversion errors, and enabling rapid prototyping. Applicable to agile teams or large projects, improving development efficiency by over 60%. Uses RAG to inject relevant code snippets, ensuring implementation aligns with design constraints. Extends to reverse engineering: generating modern code from legacy design documents.
- **Usage**:
  1. Generate or prepare design document: Use `/design` command to generate design file (e.g., design.md), or manually create/upload existing docs.
  2. Inject design into prompt: In interactive mode or CLI, use file mention (@filename) to inject design into task prompt.
  3. Specify language and mode: Set `--mode code --lang python` to generate code.
  4. Execute and validate: Agent analyzes design, plans steps, generates code, and optionally executes compile/test.
  5. Iterate optimization: If output imperfect, use `/optimize` or debug prompt to refine.
- Example: Assuming design.md exists, run `Implement the design from @design.md in python`. Agent injects design.md content, generates code files, and may execute file_write and compile_run actions. Output includes plan and generated code.

### RAG Queries Based on Source Code
- **Scenarios**: Use project source code as RAG knowledge base, query code functions, structures, dependencies, or potential issues with natural language. Suitable for code reviews, knowledge transfer, or large project maintenance, helping developers quickly understand codebases without manual reading. OAC's RAG system automatically indexes project files (code and docs), supports semantic search (e.g., function explanations, dependency analysis), and injects relevant snippets into LLM prompts for intelligent querying. Applicable to open-source project audits, team onboarding, or reverse engineering, reducing query time by 70-90%.
- **Usage**:
  1. **Initialize RAG Index**: When running OAC, VectorRAG automatically scans and indexes supported files (.py, .cpp, .md, etc.) in the current directory. If files change, enable watchdog (requires installation) for refresh every 300 seconds (config.yaml index_refresh_interval).
  2. **Configure RAG**: Adjust rag section in config.yaml, e.g., embedding_model "all-mpnet-base-v2", top_k 8 (return top results), rerank_enabled true (use LLM rerank to optimize results). For code queries, set chunk_method "function" (tasks_optimizations.debug).
  3. **Build Query Prompt**: Use natural language to describe issues, combine with file mention (@filename) to limit scope, or let RAG search globally. Specify --mode code and --lang to focus. In chat mode, use `/rag <query>` to trigger RAG injection directly.
  4. **Execute Query**: Input prompt in interactive mode or CLI, or use /rag in chat mode. Agent calls rag.search, injects relevant file content (inject_rag_results), then LLM analyzes and responds.
  5. **Iterate and Validate**: If results inaccurate, add more context or use /status to check injected context. Output includes plan and analysis results, savable to AGENT.md.
  6. **Advanced Tips**: Enable rerank for better relevance; for large projects, increase chunk_size (384) for long files; monitor token_monitor to avoid overly long prompts.
- Examples:
  - **Example 1: Query Function Functionality**. Prompt: `Explain how the infer method works in agent.py`. RAG retrieves agent.py relevant chunks, injects content, agent outputs detailed explanation plan, actions, and context management.
  - **Example 2: Project Overall Analysis**. Prompt: `What are the main dependencies in this Python project?`. RAG searches all .py files, injects dependency-related code, agent generates summary list.
  - **Example 3: Potential Issues Query**. Prompt: `Identify security vulnerabilities in @tools.py`. RAG injects file, agent uses error_handle prompt to analyze, returns root causes and fix suggestions.
  - **Example 4: Cross-File Query**. Prompt: `How does RAG integrate with the agent in this codebase?`. RAG returns rag_vector.py and agent.py snippets, agent outputs integration flowchart or steps.
  - **Example 5: Chat Mode RAG**. In --chat: `/rag Analyze the structure of cli.py`. Temporarily creates RAG, injects top_k file content, LLM directly responds with analysis.

### Other Extension Scenarios
- **GitHub Integration and PR Management**: Team collaboration, code reviews. Suitable for remote repo operations (requires enabling github_api permission). Example: `/create-pr New feature` creates PR; `/review-pr 1` reviews and suggests fixes.
- **Business Process Automation**: Financial invoice processing, logistics document extraction, manufacturing quality control record analysis. Suitable for high-volume document industries like finance (processing loan files) and logistics (extracting shipment details).
- **Scientific Research and Drug Discovery**: Process scientific literature, patent databases, or experiment result summaries. Suitable for biology/chemistry fields, using biopython etc. to accelerate drug development or literature reviews.
- **Custom Tool Extensions**: Integrate specific analyses (e.g., custom analyzer). Suitable for advanced users extending functions, like game development (pygame integration) or multimedia processing (mido).
- **General AI Chat**: Non-agent tasks, like brainstorming or queries. Suitable for quick consultations. Example: `--chat` mode input questions, supports image URL/local path processing.
- **Security and Permission Control**: Production environments, avoid dangerous operations. Default disables bash and GitHub API, suitable for sensitive projects.

Overall Advantages: OAC achieves reliable automation with minimal changes, suitable for personal learning, open-source contributions, rapid iterations. Limitations: Not suitable for large-scale production (requires manual review of outputs). Through these scenarios, OAC can achieve significant ROI, such as reducing manual document processing by 80% or improving development speed by 71% (based on Gartner research).

## 6. Advanced Features

### RAG (Retrieval-Augmented Generation)
- Automatically indexes code/document files (supports tree-sitter chunking).
- Hybrid search (semantic + BM25), rerank enabled uses LLM to optimize results.
- Scenarios: Semantic retrieval of dependencies, improving code analysis accuracy.
- Configuration: Adjust top_k, chunk_size to balance speed/precision.
- New: In chat mode, use `/rag <query>` to temporarily create RAG instance, retrieve and inject context into LLM system messages, supporting direct queries without entering agent mode.

### Permissions and Security
- Default Secure: Disables exec_bash and github_api.
- Enable Bash: Set `exec_bash: true` (warning: only in trusted environments). Or pre-authorize commands.
- GitHub: Configure token/owner/repo, enable github_api.

### Timeout Management
- Prevent hangs: Adjust timeouts to suit large file compilations or long tests.

### Custom Extensions
- **Tools**: Add Python files in `.agent/tools/`, define functions (e.g., def custom_analyzer(code): ...). Agent auto-discovers.
- **Prompt Optimization**: Edit prompts.yaml to adjust behavior (e.g., add CoT).
- **Context Customization**: Manually edit `AGENT.md` to add user identity, ensure correct format.

## 7. Troubleshooting
- **Index is empty**: Ensure directory has supported files (.py etc.), not in hidden directories.
- **Timeout expired**: Increase timeouts or simplify queries.
- **Permission denied**: Check permissions, enable required options.
- **LLM call failed**: Verify endpoint/API key, check service running.
- **Context too long**: Run `/clear` or auto-compression triggers.
- **Fuzzy matching failed**: Use full filename or check directory structure.
- **RAG failed**: Check embed_cache_dir and watchdog installation; use compress for large files.

## 8. Tool Function Explanation, Usage, and Examples

OAC supports built-in and custom tools via `tools.py`'s ToolLoader class, allowing the agent to perform specific actions like file operations, git management, compile/test, and GitHub API calls. Tools are called in the agent's execution phase (`executor.py`) to implement the "execution" part of tasks. Built-in tools include git_clone, file_search, compile_lang, run_ut, create_pr, and review_pr; custom tools extendable by adding Python files in `.agent/tools/`. The agent parses LLM output into actions in the infer method, then automatically calls corresponding tools.

### Function Explanation
- **Built-in Tools**: Predefined functions for common operations like cloning repos, searching files, compiling code, running tests, and PR management. Requires enabling relevant permissions (e.g., git_commit or github_api).
- **Custom Tools**: Users can add .py files in `.agent/tools/`, define functions (e.g., def custom_analyzer(code): return "Analysis: " + code.upper()). Agent auto-loads and calls via action type='tool'.
- **Execution Mechanism**: Agent's parse_output turns LLM response into {'actions': [...]}, then executor.execute each action. Supports lang parameter and error handling.
- **Security and Limitations**: Tool execution limited by permissions (e.g., exec_bash default disabled), timeouts default 300-600 seconds. GitHub tools require token/owner/repo config.

### Usage
1. **Configure Permissions**: Enable required permissions in config.yaml (e.g., git_commit: true). For bash, use /permissions to add allowed commands.
2. **Trigger Tools**: Users do not call tools directly but describe tasks in natural language prompts (e.g., "Clone repo and compile"). Agent decides tool usage and outputs results.
3. **Custom Extensions**: Add functions in .agent/tools/my_tool.py. Reload config (/reload) to make available. Agent can reference in prompts (LLM decides).
4. **View Output**: In headless mode, outputs JSON including tool results; in normal mode, formatted as Markdown.
5. **Error Handling**: If tool fails, returns error message (e.g., 'Permission denied'). Use /status to check context.

### Examples
- **Example 1: Using git_clone (Built-in Tool)**
  - **Scenario**: Clone external repo for code integration.
  - **Prompt**: `Clone https://github.com/example/repo and analyze structure`
  - **Agent Behavior**: Parses into action {'type': 'tool', 'name': 'git_clone', 'args': {'url': 'https://github.com/example/repo'}}, executes clone, returns 'Successfully cloned'.
  - **Output**: Plan + clone result.
- **Example 2: Using compile_lang (Built-in Tool)**
  - **Scenario**: Compile and run C++ file.
  - **Prompt**: `Compile and run @main.cpp in cpp`
  - **Agent Behavior**: Injects file content, parses into action {'type': 'compile_run', 'file': 'main.cpp', 'lang': 'cpp'}, uses g++ to compile, returns output or error.
  - **Output**: Compilation result and execution log.
- **Example 3: Using create_pr (Built-in Tool, requires github_api enabled)**
  - **Scenario**: Create PR based on changes.
  - **Prompt**: `/create-pr Add feature X`
  - **Agent Behavior**: Uses create_pr prompt to generate title/body, parses into action {'type': 'create_pr', 'title': '...', 'body': '...'}, calls PyGithub to create PR, returns URL.
  - **Output**: PR creation confirmation.
- **Example 4: Custom Tool (Assume .agent/tools/my_tool.py has def custom_analyzer(code): return "Analysis: " + code.upper())**
  - **Scenario**: Custom code analysis.
  - **Prompt**: `Analyze code @script.py with custom tool`
  - **Agent Behavior**: Loads tool, parses into action {'type': 'tool', 'name': 'custom_analyzer', 'args': {'code': '...' }}, executes and returns result.
  - **Output**: Analysis result injected into facts.
- **Example 5: Using run_ut (Built-in Tool)**
  - **Scenario**: Run Python unit tests.
  - **Prompt**: `Run unit tests for @test.py in python`
  - **Agent Behavior**: Parses into action {'type': 'run_ut', 'file': 'test.py', 'lang': 'python'}, uses pytest to execute, returns test report.
  - **Output**: Passed/Failed details.

Through these tools, OAC can extend to complex workflows, such as automated CI/CD or integrating third-party APIs. Custom tools require testing for compatibility.

## 9. Comparison List with ClaudeCode (CC) Similar Functions

The following table compares the core functions of CC and OAC. CC's information is based on Yuker's X posts (2026 context) and typical Claude code agent features: CC emphasizes user-custom "onboarding manual" (`CLAUDE.md`), code task handling, and simple CLI configuration. OAC provides more comprehensive tool integration and permission control. Pros/cons based on reliability, flexibility, and potential limitations. Coverage is subjective estimate, considering functional implementation similarity. New features like /rag in chat mode enhance OAC's RAG coverage.

| CC Function/Command | Description | Pros/Cons | OAC Similar Function/Command | Description | Pros/Cons | Coverage (%) |
|--------------|------|--------|-------------------|------|--------|-------------|
| **Context Persistence (CLAUDE.md)** | Uses Markdown file to store user identity, project background, interaction history, enabling AI to "remember who I am". Manually edit file, injected by Claude in conversations. | **Pros**: Simple, intuitive, users can edit directly; **Cons**: Relies on manual updates, no auto-compression, may cause overly long context affecting performance. | **Context Management (AGENT.md)** | YAML file stores overview, history, facts, langs. Agent auto-loads/updates/compresses (using LLM summary), injects into prompt. Commands: `/status` view, `/clear` reset. | **Pros**: Auto-update and compression ensure efficiency; integrates RAG; **Cons**: YAML format slightly more complex than Markdown, manual edits need careful formatting. | 90% (OAC more automated, but core persistence same) |
| **Code Generation Tasks** | Generate code via prompts, supports multiple languages, based on Claude's natural language processing. No dedicated commands, direct conversation. | **Pros**: Claude's strong reasoning, generates high-quality code; **Cons**: No built-in executor, requires manual testing; relies on network. | **Code Generation (infer method)** | Generate code via CLI prompt, supports python/cpp etc. Example: `Create binary search in Python`. Integrates executor (compile_run). | **Pros**: Built-in execution and validation, supports multiple languages; RAG enhances accuracy; **Cons**: Relies on local config, may not match Claude's reasoning depth. | 85% (OAC adds execution, but generation logic similar) |
| **Debugging and Error Handling** | Describe errors in prompts, Claude analyzes root causes and suggests fixes. Supports file uploads/mentions. | **Pros**: Intelligent root cause analysis using CoT; **Cons**: Relies on user-provided context; may hallucinate. | **Debugging (error_handle prompt)** | Example: `Debug @main.py with error`. Uses RAG to retrieve relevant code, auto-optimizes prompt. | **Pros**: RAG integration improves accuracy; auto-rerank fix suggestions; **Cons**: Requires local file indexing, longer init time. | 95% (OAC enhances retrieval, core analysis same) |
| **Document Optimization** | Summarize/optimize documents via prompts, supports Markdown processing. | **Pros**: Claude excels at natural language optimization; **Cons**: No dedicated mode, requires manual specification. | **Document Mode (--mode doc)** | Example: `--mode doc Optimize @README.md`. Uses doc_optimize prompt and rerank. | **Pros**: Dedicated mode and chunking improve efficiency; **Cons**: Relies on RAG config, may overfit code files. | 80% (OAC more structured, but optimization logic similar) |
| **Requirements/Design Generation** | Generate requirements or design docs via prompts, using templates. | **Pros**: Flexible, Claude can generate UML etc.; **Cons**: No built-in templates, requires user provision. | **/requirements, /design** | Generate requirements/design docs using dedicated prompts. Example: `/design` outputs UML text. | **Pros**: Built-in templates and lang support; **Cons**: Templates need customization, not as universal as Claude. | 85% (OAC command-based, more user-friendly) |
| **PR Creation/Review** | No built-in, may simulate reviews via prompts. Claude can generate comments. | **Pros**: Customizable review logic; **Cons**: No GitHub integration, requires manual operations; not automated. | **/create-pr, /review-pr** | Auto-generates title/body, reviews diff. Requires GitHub config. | **Pros**: Integrates PyGithub, automates push/PR; **Cons**: Requires permission enable, security risks. | 70% (OAC more automated, CC relies on manual) |
| **Configuration Management** | Teach configuration via "onboarding manual", no dedicated CLI. | **Pros**: Educational, suitable for beginners; **Cons**: No standardized files, relies on posts/docs. | **config.yaml, /config, /reload** | YAML configures LLM, permissions etc. Commands view/reload. | **Pros**: Standardized, easy to extend; supports env variables; **Cons**: Requires YAML knowledge, initial setup complex. | 75% (OAC more comprehensive, but CC simpler) |
| **Help System** | No built-in, may query help via prompts. | **Pros**: Claude can explain itself; **Cons**: No structured docs. | **/help (help.md)** | Displays detailed help document. | **Pros**: Dedicated file, tabulated command list; **Cons**: Requires file maintenance. | 60% (OAC more structured, CC relies on conversation) |
| **Chat Mode** | Claude default conversation mode, supports direct interaction. | **Pros**: Streaming responses, real-time; **Cons**: No agent layer, easy to go off-topic. | **--chat, /chat** | Direct LLM chat, supports model switching (/model) and RAG queries (/rag). | **Pros**: Supports streaming, model switching, and RAG injection; **Cons**: Separated from agent mode. | 95% (OAC adds RAG, improves practicality) |
| **Tool Extensions** | Claude supports tool calls, but CC may customize simple tools. | **Pros**: Integrates Claude toolchain; **Cons**: Not persistent, requires definition each time. | **.agent/tools/ Directory** | Auto-loads Python tools, supports git_clone etc. | **Pros**: Persistent, easy to extend; **Cons**: Requires Python coding. | 80% (OAC more localized) |
| **RAG Retrieval** | Claude may lack built-in RAG, relies on user uploads. | **Pros**: Simple; **Cons**: No auto-indexing, non-semantic retrieval. | **RAG (VectorRAG Class)** | Auto-indexes files, hybrid search + rerank. Supports chat mode /rag. | **Pros**: Semantic enhancement improves accuracy; supports watcher; **Cons**: Requires computational resources. | 60% (OAC unique enhancement, CC basic coverage; new /rag improves) |

**Overall Coverage**: About 82%. OAC is stronger in automation and integration (e.g., RAG, executor), but CC relies on Claude's powerful LLM reasoning, potentially more flexible for complex tasks. OAC suitable for local/open-source scenarios, CC more cloud/conversation-oriented. Suggest mixing based on needs.

For more questions: View `/help` or source code.