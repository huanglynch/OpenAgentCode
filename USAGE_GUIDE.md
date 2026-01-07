# OpenAgentCode (OAC) Usage Guide

OpenAgentCode (OAC) is an AI-based agent tool designed to assist developers with development tasks. It integrates large language models (LLM), retrieval-augmented generation (RAG), context management, and various executors, supporting multiple programming languages (such as Python, C++, Java, etc.). OAC operates via a CLI interface, allowing users to generate code, debug errors, optimize documents, generate requirements documents, design architectures, create/review PRs, and more. The core goal is to automate software development processes, improve efficiency, while maintaining security and configurability.

OAC's design principles are based on first principles: minimal changes, simplicity and reliability, purpose-oriented. It uses YAML configuration, Markdown context files, and prompt templates to ensure easy extensibility. Suitable for individual developers, team collaboration, and small-scale projects.

## 1. Installation and Initialization

### Requirements
- Python 3.8+
- Dependencies: requests, json, yaml, click, difflib, etc. (install via pip)
- LLM Service: Supports local (e.g., Ollama) or remote APIs (e.g., xAI's Grok), requires API key configuration
- Optional: GitPython (for git operations), PyGithub (for GitHub API), sentence-transformers (for RAG embeddings), tree-sitter (for code chunking), watchdog (for file monitoring)

### Initialize Project
Run in the project directory:
```bash
python cli.py
```

If `config.yaml` does not exist, it will automatically initialize:
- Create `config.yaml` (default configuration, including LLM, paths, permissions, RAG, etc.)
- Create `prompts.yaml` (prompt templates)
- Create `AGENT.md` (context file for persisting agent "memory")
- Create `help.md` (help documentation)
- Create `.agent/tools/` directory (for custom tools)

Edit `config.yaml` to customize LLM endpoint, API key, etc. For example:
```yaml
llm:
  api_key: $YOUR_API_KEY$
  endpoint: https://api.x.ai/v1/chat/completions
  model: grok-4-1-fast-non-reasoning
```

## 2. Configuration Explanation

OAC configures all functions via `config.yaml`. Key sections:

- **llm**: Endpoint, model, temperature, max tokens, API key. Supports environment variables (e.g., `$XAI_API_KEY$`). Includes support for vision_model and endpoint_vision for image processing
- **paths**: Prompts file (`prompts.yaml`), context file (`AGENT.md`), tools directory (`.agent/tools/`), embedding cache (`cached_models/`), help file (`help.md`)
- **permissions**: File read/write, git commit, bash execution, GitHub API. Default secure settings (bash and GitHub API disabled). Can pre-authorize bash commands
- **modes**: Default `code` or `doc`
- **timeouts**: Bash execution, compilation, unit tests, LLM requests, tool execution (default 300-600 seconds, adjustable for large projects)
- **rag**: Embedding model (`all-mpnet-base-v2`), hybrid alpha (0.4), top_k (8), chunk_size (384), rerank_enabled (true), refresh interval (300 seconds)
- **languages**: Supports python, cpp, c, js, java, cs, go; default python
- **workspace**: Current working directory path for setting agent work environment. Dynamically switchable and persistent via `/workspace` command
- **tasks_optimizations**: Optimization templates for debug, ut, doc, requirements, design, optimize
- **github**: Token, owner, repo (for PR operations)

**Prompt Templates**: Customize base_prompt, error_handle, etc., in `prompts.yaml` to adjust agent behavior.

**Context Management**: `AGENT.md` stores overview, history, facts, langs. The agent automatically loads/updates/compresses to ensure "remembering" user identity and history.

**Reload Configuration**: Run `/reload` to apply changes.

## 3. Usage Methods

### CLI Direct Commands
```bash
python cli.py "Your task here" [OPTIONS]
```

Options:
- `--mode TEXT`: code or doc (default: code)
- `--headless`: Output JSON only
- `--lang TEXT`: Language (e.g., python, cpp)
- `--chat`: Enter direct chat mode (without agent)

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
- Input tasks or slash commands
- Supports file mentions: `@filename` (fuzzy matching, injects file content)
- Exit: `/exit` or `/quit`

### Chat Mode
```bash
python cli.py --chat
```

Direct conversation with LLM, supports streaming output. Commands: `/model endpoint_type,model_name` to switch models; `/rag <query>` supports RAG queries (retrieves local files and injects into context).

### Multimodal Support (Image Processing)
- **Description**: In --chat mode, OAC supports sending image files or URLs to vision-capable LLMs for analysis, description, or recognition. Suitable for image-related queries like object recognition, text extraction, or visual descriptions.

- **Usage**:
  1. Enter --chat mode: `python cli.py --chat`
  2. Input query with image:
     - **URL**: e.g., "What is in this picture? https://example.com/cat.jpg". OAC automatically detects and builds multimodal message
     - **Local Path**: e.g., "Describe: D:/path/to/photo.png". OAC reads file, converts to base64 data URI for sending
  3. Supported Formats: .jpg, .jpeg, .png, .gif
  4. Configuration: Ensure llm.vision_model and llm.endpoint_vision are correct in config.yaml (default uses xAI API)

- **Example**:
  - Input: "Analyze this image: http://example.com/chart.png"
  - Output: AI response like "The image shows a bar chart with sales data..."

- **Limitations**: Limited to --chat mode; RAG does not index images; does not support image editing or generation (extendable via custom tools). Prints error if read fails.

## 4. Command List

### Slash Commands (In interactive mode or direct prompt starting with `/`)

| Command | Description | Simple Example |
|---------|-------------|----------------|
| `/help` | Display help information (loaded from `help.md`) | `/help` → Outputs full help document |
| `/exit`, `/quit` | Exit interactive mode | `/exit` → "Goodbye!" |
| `/clear` | Clear agent context (`AGENT.md`) | `/clear` → "Context cleared." |
| `/chat` | Enter chat mode | `/chat` → "You: " prompt |
| `/status` | Display context overview (overview, history entries, facts, langs) | `/status` → "History entries: 5" |
| `/config` | Display current configuration (YAML format, API key placeholder) | `/config` → Outputs config.yaml content |
| `/reload` | Reload config.yaml and prompts.yaml | `/reload` → "Configuration and prompts reloaded." |
| `/create-pr [message]` | Create PR (generate title, body based on changes). Requires GitHub config | `/create-pr Add new feature` → Creates PR and outputs URL |
| `/review-pr [number]` | Review PR (analyze diff, check bugs, security, style) | `/review-pr 5` → Outputs review comments JSON |
| `/model [endpoint_type,model_name]` | Switch LLM model and endpoint (e.g., vllm,AI:Pro). Saves to config.yaml | `/model ollama,qwen3:1.7b` → "Switched to model qwen3:1.7b" |
| `/permissions [command]` | List or add allowed bash commands (if exec_bash disabled) | `/permissions git push` → Adds and saves. `/permissions` → Lists allowed commands |
| `/workspace [path]` | Switch working directory and persist to config file. Supports relative/absolute paths, automatically rebuilds RAG index | `/workspace ./my-project` → Switches to specified directory and saves setting |
| `/commit-push-pr [message]` | Commit changes, push branch, create PR | `/commit-push-pr Fix bug` → Executes git operations and creates PR |
| `/requirements` | Generate requirements.md (functional/non-functional requirements, constraints) | `/requirements` → Outputs requirements document |
| `/design` | Generate design document (UML text, components, architecture) | `/design` → Outputs design document |
| `/optimize` | Optimize code or documents (performance, readability, refactoring suggestions) | `/optimize` → Outputs optimization suggestions |
| `/rag <query>` | (Chat mode only) Use RAG to retrieve local files and inject into LLM query | `/rag Explain the infer method in agent.py` → Outputs RAG-based response |

### Other Task Commands (Non-slash, direct prompt)
- Starting with `/` but not listed above: Treated as tasks (e.g., `/some-task` → Agent infers)

## 5. Application Scenarios

OAC is suitable for the full software development lifecycle, especially scenarios needing AI assistance. The following are categorized by function, with expanded application suggestions based on real industry use cases, such as automated document processing (reducing 70-85% processing costs), literary optimization (accelerating creative iterations), design generation (architecture planning), requirements analysis (drafting user stories), ROI reports (quantifying tool benefits), and others like business automation, scientific research, and manufacturing, to provide comprehensive guidance.

### Code Generation and Optimization
- **Scenarios**: Rapid prototyping, algorithm implementation, code refactoring. Suitable for beginners or time-constrained projects
- **Example**: Generate sorting algorithms, optimize performance bottlenecks. Command: `Implement quicksort in cpp`

### Debugging and Error Handling
- **Scenarios**: Track bugs, analyze logs. Suitable for error diagnosis in complex projects
- **Example**: `Debug @main.py with error`. Agent uses RAG to retrieve relevant snippets and inject context

### Unit Testing (UT)
- **Scenarios**: Automated test generation and execution. Suitable for TDD (Test-Driven Development) or CI/CD pipelines
- **Example**: `Generate unit tests for @calculator.py`. Supports frameworks like pytest

### Document Management and Processing/Editing/Optimization
- **Scenarios**: Summarize README, optimize document clarity, automate data extraction and classification. Suitable for open-source projects, team collaboration, or high-volume document processing (e.g., invoices, contracts, reports). OAC can handle PDF/text files with 99%+ accuracy in data extraction, reducing manual work by 80%, and ensuring compliance audits. Extends to enterprise scenarios: processing financial reports, legal documents, or compliance docs, generating summaries or optimizing redundant content
- **Example**: `--mode doc Summarize @README.md` or `Optimize contract @legal_doc.md`. Uses doc_optimize prompt to reduce redundancy

### Literary Creation/Optimization
- **Scenarios**: Generate story outlines, optimize novel/article structures, summarize literary works, or generate creative content. Suitable for writers, content creators, or education, helping accelerate iterations (e.g., from draft to refined version), or analyze literary themes. OAC can inject context history for continuous creation and use RAG to retrieve similar works to enhance originality. Applicable to blog optimization, script writing, or academic paper polishing, reducing creation time by 50%
- **Example**: `Generate a short story outline about AI agents` or `Optimize this novel chapter @draft.txt for clarity and flow`

### Architecture and System Design
- **Scenarios**: Software architecture design, UI/UX prototype generation, threat modeling, or service boundary planning. Suitable for architects or product teams, generating UML diagrams, ADR (Architecture Decision Records), or performance budgets from requirements. Extends to non-software design: such as generating product design documents, flowcharts, or hardware architecture suggestions. OAC can proactively suggest options, pros/cons, and integrate existing codebases for end-to-end design automation
- **Example**: `/design Generate architecture for a user management system` outputs UML text and component diagrams

### Requirements Analysis
- **Scenarios**: Generate requirements documents, user stories, priority sorting, or non-functional requirements (NFRs) from user feedback. Suitable for product managers in project initiation phases, helping process feedback aggregation, deduplication, and acceptance criteria suggestions. Extends to market research: analyzing competitor product requirements or extracting key requirements from literature, accelerating discovery phases
- **Example**: `/requirements Analyze user feedback for a mobile app` outputs functional/non-functional requirements and constraints

### ROI Analysis and Reporting
- **Scenarios**: Quantify ROI for AI tools or projects, such as tracking code generation speed, debugging time reduction, document efficiency improvements. Suitable for engineering leaders evaluating tool adoption frameworks (e.g., Gartner model), generating reports including time savings, productivity gains (average 43% efficiency increase), and cost reductions ($2.3M/year). OAC can use tool extensions to generate charts or summarize data, supporting decisions like integrating new AI agents
- **Example**: `Generate ROI report for implementing OAC in our dev team` outputs analysis and suggestions

### Code Implementation from Design Documents
- **Scenarios**: Automatically generate initial code frameworks or complete implementations from existing design documents (e.g., UML, architecture descriptions). Suitable for software development iteration phases, bridging design and coding, reducing manual conversion errors, and enabling rapid prototyping. Applicable to agile teams or large projects, improving development efficiency by over 60%. Uses RAG to inject relevant code snippets, ensuring implementation aligns with design constraints
- **Usage**:
  1. Generate or prepare design document: Use `/design` command to generate design file (e.g., design.md), or manually create/upload existing docs
  2. Inject design into prompt: In interactive mode or CLI, use file mention (@filename) to inject design into task prompt
  3. Specify language and mode: Set `--mode code --lang python` to generate code
  4. Execute and validate: Agent analyzes design, plans steps, generates code, and optionally executes compile/test
  5. Iterate optimization: If output imperfect, use `/optimize` or debug prompt to refine
- **Example**: Assuming design.md exists, run `Implement the design from @design.md in python`. Agent injects design.md content, generates code files, and may execute file_write and compile_run actions

### Intelligent Code Base Analysis via RAG
- **Scenarios**: Use project source code as RAG knowledge base, query code functions, structures, dependencies, or potential issues with natural language. Suitable for code reviews, knowledge transfer, or large project maintenance, helping developers quickly understand codebases without manual reading. OAC's RAG system automatically indexes project files (code and docs), supports semantic search (e.g., function explanations, dependency analysis), and injects relevant snippets into LLM prompts for intelligent querying. Applicable to open-source project audits, team onboarding, or reverse engineering, reducing query time by 70-90%
- **Usage**:
  1. **Initialize RAG Index**: When running OAC, VectorRAG automatically scans and indexes supported files (.py, .cpp, .md, etc.) in the current directory. If files change, enable watchdog (requires installation) for refresh every 300 seconds
  2. **Configure RAG**: Adjust rag section in config.yaml, e.g., embedding_model "all-mpnet-base-v2", top_k 8 (return top results), rerank_enabled true (use LLM rerank to optimize results)
  3. **Build Query Prompt**: Use natural language to describe issues, combine with file mention (@filename) to limit scope, or let RAG search globally. Specify --mode code and --lang to focus. In chat mode, use `/rag <query>` to trigger RAG injection directly
  4. **Execute Query**: Input prompt in interactive mode or CLI, or use /rag in chat mode. Agent calls rag.search, injects relevant file content, then LLM analyzes and responds
  5. **Iterate and Validate**: If results inaccurate, add more context or use /status to check injected context. Output includes plan and analysis results, savable to AGENT.md
- **Examples**:
  - **Function Analysis**: `Explain how the infer method works in agent.py` → RAG retrieves relevant chunks, agent provides detailed explanation
  - **Dependency Analysis**: `What are the main dependencies in this Python project?` → Agent generates comprehensive dependency summary
  - **Security Audit**: `Identify security vulnerabilities in @tools.py` → Agent analyzes and returns security recommendations
  - **Integration Analysis**: `How does RAG integrate with the agent in this codebase?` → Agent outputs integration flowchart
  - **Chat Mode RAG**: `/rag Analyze the structure of cli.py` → Direct analysis without entering agent mode

### Extended Scenarios
- **GitHub Integration and PR Management**: Team collaboration, code reviews. Suitable for remote repo operations (requires enabling github_api permission). Example: `/create-pr New feature` creates PR; `/review-pr 1` reviews and suggests fixes
- **Multi-Project Management**: Use `/workspace` command to quickly switch between different project directories, each maintaining independent RAG indexes. Suitable for developers maintaining multiple code repositories, enabling rapid project switching without reinitializing configuration. Example: `/workspace ~/project-a` → `/workspace ~/project-b`, each switch automatically rebuilds corresponding project code index
- **Business Process Automation**: Financial invoice processing, logistics document extraction, manufacturing quality control record analysis. Suitable for high-volume document industries like finance (processing loan files) and logistics (extracting shipment details)
- **Scientific Research and Analysis**: Process scientific literature, patent databases, or experiment result summaries. Suitable for biology/chemistry fields, using specialized libraries to accelerate research or literature reviews
- **Custom Tool Integration**: Integrate domain-specific analysis tools. Suitable for advanced users extending functionality for specialized workflows
- **Interactive AI Assistance**: Non-agent tasks, like brainstorming or general queries. Suitable for quick consultations. Example: `--chat` mode supports both text and image inputs
- **Security and Compliance**: Production environments requiring controlled operations. Default security settings disable potentially dangerous operations, suitable for enterprise environments

**Overall Advantages**: OAC achieves reliable automation with minimal setup complexity, suitable for personal learning, open-source contributions, and rapid development iterations. **Limitations**: Requires manual review for production deployment. Through these scenarios, OAC can achieve significant ROI, such as reducing manual document processing by 80% or improving development speed by 71%.

## 6. Advanced Features

### RAG (Retrieval-Augmented Generation)
- Automatically indexes code/document files (supports tree-sitter chunking for better code understanding)
- Hybrid search (semantic + BM25), rerank enabled uses LLM to optimize results for relevance
- **Scenarios**: Semantic retrieval of dependencies, improving code analysis accuracy, intelligent project exploration
- **Configuration**: Adjust top_k, chunk_size to balance speed/precision for your project size
- **New Feature**: In chat mode, use `/rag <query>` to temporarily create RAG instance, retrieve and inject context into LLM system messages, supporting direct queries without entering agent mode

### Permissions and Security
- **Default Secure**: Disables exec_bash and github_api for safety
- **Enable Bash**: Set `exec_bash: true` (⚠️ Warning: only in trusted environments) or pre-authorize specific commands
- **GitHub Integration**: Configure token/owner/repo, enable github_api for automated PR workflows

### Timeout Management
- **Purpose**: Prevent system hangs during long operations
- **Configuration**: Adjust timeouts in config.yaml to suit large file compilations, extensive tests, or complex queries

### Custom Extensions
- **Tools**: Add Python files in `.agent/tools/`, define functions (e.g., `def custom_analyzer(code): ...`). Agent auto-discovers and integrates
- **Prompt Optimization**: Edit prompts.yaml to adjust agent behavior (e.g., add Chain-of-Thought reasoning)
- **Context Customization**: Manually edit `AGENT.md` to add user identity and project-specific information, ensuring correct YAML format

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| **Index is empty** | Ensure directory contains supported files (.py, .cpp, .md, etc.) and is not in hidden directories |
| **Timeout expired** | Increase timeout values in config.yaml or simplify complex queries |
| **Permission denied** | Check and enable required permissions in config.yaml |
| **LLM call failed** | Verify endpoint URL and API key configuration, ensure service is running |
| **Context too long** | Run `/clear` to reset context or allow auto-compression to trigger |
| **Fuzzy matching failed** | Use complete filename or verify directory structure |
| **RAG indexing failed** | Check embed_cache_dir permissions and watchdog installation; large files may need compression |

## 8. Tool System: Functions, Usage, and Examples

OAC supports built-in and custom tools via the `tools.py` ToolLoader class, enabling the agent to perform specific actions like file operations, git management, compilation/testing, and GitHub API integration. Tools are executed during the agent's action phase in `executor.py`, implementing the practical "execution" component of tasks. Built-in tools include git_clone, file_search, compile_lang, run_ut, create_pr, and review_pr. Custom tools are extensible by adding Python files to `.agent/tools/`. The agent parses LLM output into structured actions in the infer method, then automatically invokes corresponding tools.

### Function Overview
- **Built-in Tools**: Predefined functions for common development operations like repository cloning, file searching, code compilation, test execution, and PR management. Requires enabling relevant permissions (e.g., git_commit or github_api)
- **Custom Tools**: Users can add .py files in `.agent/tools/` directory, defining functions (e.g., `def custom_analyzer(code): return "Analysis: " + code.upper()`). Agent automatically loads and calls via action type='tool'
- **Execution Mechanism**: Agent's parse_output converts LLM response into `{'actions': [...]}` format, then executor.execute processes each action. Supports language parameters and comprehensive error handling
- **Security and Limitations**: Tool execution governed by permission system (e.g., exec_bash default disabled), timeouts default 300-600 seconds. GitHub tools require token/owner/repo configuration

### Usage Methodology
1. **Configure Permissions**: Enable required permissions in config.yaml (e.g., `git_commit: true`). For bash operations, use `/permissions` to add specific allowed commands
2. **Tool Activation**: Users describe tasks in natural language prompts rather than calling tools directly (e.g., "Clone repo and compile"). Agent determines appropriate tool usage and provides results
3. **Custom Extensions**: Add functions to `.agent/tools/my_tool.py`. Use `/reload` to make tools available. Agent references tools in prompts based on LLM decision-making
4. **Output Viewing**: Headless mode outputs JSON including tool results; normal mode formats as readable Markdown
5. **Error Handling**: Failed tools return descriptive error messages (e.g., 'Permission denied'). Use `/status` to inspect current context

### Practical Examples
- **Example 1: Repository Cloning (Built-in Tool)**
  - **Scenario**: Clone external repository for code integration
  - **Prompt**: `Clone https://github.com/example/repo and analyze structure`
  - **Agent Behavior**: Parses into action `{'type': 'tool', 'name': 'git_clone', 'args': {'url': 'https://github.com/example/repo'}}`, executes clone operation, returns 'Successfully cloned'
  - **Output**: Execution plan + clone results

- **Example 2: Code Compilation (Built-in Tool)**
  - **Scenario**: Compile and execute C++ file
  - **Prompt**: `Compile and run @main.cpp in cpp`
  - **Agent Behavior**: Injects file content, parses into action `{'type': 'compile_run', 'file': 'main.cpp', 'lang': 'cpp'}`, uses g++ compiler, returns output or error messages
  - **Output**: Compilation results and execution logs

- **Example 3: PR Creation (Built-in Tool, requires github_api enabled)**
  - **Scenario**: Create pull request based on current changes
  - **Prompt**: `/create-pr Add feature X`
  - **Agent Behavior**: Uses create_pr prompt template to generate title/body, parses into action `{'type': 'create_pr', 'title': '...', 'body': '...'}`, calls PyGithub API to create PR, returns URL
  - **Output**: PR creation confirmation with link

- **Example 4: Custom Analysis Tool**
  - **Setup**: Create `.agent/tools/my_tool.py` with `def custom_analyzer(code): return "Analysis: " + code.upper()`
  - **Scenario**: Custom code analysis workflow
  - **Prompt**: `Analyze code @script.py with custom tool`
  - **Agent Behavior**: Loads custom tool, parses into action `{'type': 'tool', 'name': 'custom_analyzer', 'args': {'code': '...'}}`, executes and returns custom analysis result
  - **Output**: Analysis result integrated into facts

- **Example 5: Unit Testing (Built-in Tool)**
  - **Scenario**: Execute Python unit tests
  - **Prompt**: `Run unit tests for @test.py in python`
  - **Agent Behavior**: Parses into action `{'type': 'run_ut', 'file': 'test.py', 'lang': 'python'}`, uses pytest framework, returns comprehensive test report
  - **Output**: Detailed pass/fail results with coverage information

Through this tool system, OAC extends to complex workflows including automated CI/CD pipelines, third-party API integration, and domain-specific analysis tools. Custom tools require compatibility testing with the agent framework.

## 9. Comparison with ClaudeCode (CC) Features

The following table compares core functionalities between CC and OAC. CC information is based on Yuker's X posts (2026 context) and typical Claude code agent characteristics: CC emphasizes user-customizable "onboarding manual" (`CLAUDE.md`), code task processing, and simple CLI configuration. OAC provides more comprehensive tool integration and granular permission control. Pros/cons evaluated based on reliability, flexibility, and operational limitations. Coverage percentages represent subjective estimates considering functional implementation similarity. New features like `/rag` in chat mode significantly enhance OAC's RAG capabilities.

| CC Function/Command | Description | Pros/Cons | OAC Equivalent Function/Command | Description | Pros/Cons | Coverage (%) |
|---------------------|-------------|-----------|--------------------------------|-------------|-----------|--------------|
| **Context Persistence (CLAUDE.md)** | Uses Markdown file to store user identity, project background, interaction history, enabling AI to "remember who I am". Manual file editing, Claude injects during conversations | **Pros**: Simple, intuitive, direct user editing; **Cons**: Manual update dependency, no auto-compression, potential context length issues affecting performance | **Context Management (AGENT.md)** | YAML file stores overview, history, facts, langs. Agent automatically loads/updates/compresses (using LLM summarization), injects into prompts. Commands: `/status` view, `/clear` reset | **Pros**: Automated update/compression ensures efficiency; RAG integration; **Cons**: YAML format slightly more complex, manual edits require careful formatting | 90% (OAC more automated, core persistence equivalent) |
| **Code Generation Tasks** | Generate code via prompts, supports multiple languages, leverages Claude's natural language processing. No dedicated commands, direct conversation | **Pros**: Claude's powerful reasoning generates high-quality code; **Cons**: No built-in executor, manual testing required; network dependency | **Code Generation (infer method)** | Generate code via CLI prompts, supports python/cpp etc. Example: `Create binary search in Python`. Integrates executor (compile_run) | **Pros**: Built-in execution and validation, multi-language support; RAG enhances accuracy; **Cons**: Local configuration dependency, may not match Claude's reasoning depth | 85% (OAC adds execution capability, similar generation logic) |
| **Debugging and Error Handling** | Describe errors in prompts, Claude analyzes root causes and suggests fixes. Supports file uploads/mentions | **Pros**: Intelligent root cause analysis using Chain-of-Thought; **Cons**: User-provided context dependency; potential hallucination issues | **Debugging (error_handle prompt)** | Example: `Debug @main.py with error`. Uses RAG to retrieve relevant code, auto-optimizes prompts | **Pros**: RAG integration improves accuracy; auto-rerank fix suggestions; **Cons**: Local file indexing required, longer initialization time | 95% (OAC enhances retrieval, equivalent core analysis) |
| **Document Optimization** | Summarize/optimize documents via prompts, supports Markdown processing | **Pros**: Claude excels at natural language optimization; **Cons**: No dedicated mode, manual specification required | **Document Mode (--mode doc)** | Example: `--mode doc Optimize @README.md`. Uses doc_optimize prompt and rerank | **Pros**: Dedicated mode and chunking improve efficiency; **Cons**: RAG configuration dependency, potential code file overfitting | 80% (OAC more structured, similar optimization logic) |
| **Requirements/Design Generation** | Generate requirements or design documents via prompts, using templates | **Pros**: Flexible, Claude can generate UML etc.; **Cons**: No built-in templates, user provision required | **/requirements, /design** | Generate requirements/design documents using dedicated prompts. Example: `/design` outputs UML text | **Pros**: Built-in templates and language support; **Cons**: Template customization needed, less universal than Claude | 85% (OAC command-based, more user-friendly) |
| **PR Creation/Review** | No built-in functionality, may simulate reviews via prompts. Claude can generate comments | **Pros**: Customizable review logic; **Cons**: No GitHub integration, manual operations required; not automated | **/create-pr, /review-pr** | Auto-generates title/body, reviews diffs. Requires GitHub configuration | **Pros**: PyGithub integration, automates push/PR workflows; **Cons**: Permission enablement required, security considerations | 70% (OAC more automated, CC manual dependency) |
| **Configuration Management** | Teach configuration via "onboarding manual", no dedicated CLI | **Pros**: Educational approach, beginner-friendly; **Cons**: No standardized files, relies on posts/documentation | **config.yaml, /config, /reload** | YAML configures LLM, permissions etc. Commands for viewing/reloading | **Pros**: Standardized, easily extensible; environment variable support; **Cons**: YAML knowledge required, complex initial setup | 75% (OAC more comprehensive, CC simpler) |
| **Help System** | No built-in help, may query assistance via prompts | **Pros**: Claude can explain itself contextually; **Cons**: No structured documentation | **/help (help.md)** | Displays detailed help documentation | **Pros**: Dedicated file, tabulated command reference; **Cons**: File maintenance required | 60% (OAC more structured, CC conversation-dependent) |
| **Chat Mode** | Claude default conversation mode, supports direct interaction | **Pros**: Streaming responses, real-time interaction; **Cons**: No agent layer, easy topic drift | **--chat, /chat** | Direct LLM chat, supports model switching (/model) and RAG queries (/rag) | **Pros**: Streaming support, model switching, RAG injection capabilities; **Cons**: Separated from agent mode | 95% (OAC adds RAG, improves practical utility) |
| **Tool Extensions** | Claude supports tool calls, CC may customize simple tools | **Pros**: Integrates Claude toolchain; **Cons**: Not persistent, requires redefinition | **.agent/tools/ Directory** | Auto-loads Python tools, supports git_clone etc. | **Pros**: Persistent, easily extensible; **Cons**: Python coding required | 80% (OAC more localized approach) |
| **RAG Retrieval** | Claude may lack built-in RAG, relies on user uploads | **Pros**: Simplicity; **Cons**: No auto-indexing, non-semantic retrieval | **RAG (VectorRAG Class)** | Auto-indexes files, hybrid search + rerank. Supports chat mode /rag | **Pros**: Semantic enhancement improves accuracy; watcher support; **Cons**: Computational resource requirements | 60% (OAC unique enhancement, CC basic coverage; /rag feature significantly improves coverage) |

**Overall Coverage**: Approximately 82%. OAC demonstrates superior automation and integration capabilities (e.g., RAG, executor), while CC leverages Claude's powerful LLM reasoning, potentially offering more flexibility for complex tasks. OAC is optimized for local/open-source scenarios, while CC is more cloud/conversation-oriented. Recommend hybrid usage based on specific requirements.

---

For additional information: View `/help` or consult source code documentation.
