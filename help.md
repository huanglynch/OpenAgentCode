# 文件: D:/huang/data/working/python/OpenAgentCode\help.md
# OpenAgentCode - Help Documentation
## Quick Start
### Interactive Mode
```bash
python cli.py
```
### Direct Command
```bash
python cli.py "Your task here"
```
## Commands Reference
### Slash Commands
| Command | Description |
|---------|-------------|
| `/help` | Display this help message |
| `/exit`, `/quit` | Exit interactive mode |
| `/clear` | Clear agent context history |
| `/status` | Show current context status |
| `/config` | Display current configuration |
| `/requirements` | Generate requirements.md |
| `/design` | Generate design document |
| `/optimize` | Optimize code or documentation |
| `/create-pr` | Create Pull Request |
| `/review-pr` | Review Pull Request |
| `/model [endpoint_type,model_name]` | Switch to the specified LLM endpoint and model (e.g., /model vllm,AI:Pro). |
| `/permissions [command]` | List or add allowed bash commands (if exec_bash is false) |
| `/commit-push-pr [message]` | Commit, push, and create PR |
### File Mentions
Use `@filename` to include file contents:
```
Debug @main.py with error from @error.log
Refactor @utils.py to improve performance
```
**Fuzzy Matching:**
- `@main` → matches `main.py`, `main.cpp`, etc.
- `@config` → matches `config.yaml`, `config.json`, etc.
## CLI Options
```bash
python cli.py [PROMPT] [OPTIONS]
Options:
  --mode TEXT Mode: code or doc (default: code)
  --headless Output JSON only
  --lang TEXT Language: python, cpp, c, js, java, cs, go
  --help Show CLI help
```
## Configuration
### config.yaml Structure
```yaml
llm:
  endpoint: "http://localhost:8000/api/chat"
  model: "llama-3-70b"
  temperature: 0.7
  max_tokens: 8192
  api_key: ""
timeouts:
  bash_exec: 300 # Bash command timeout (seconds)
  compile: 300 # Large C++ projects
  unit_test: 300 # Comprehensive test suites
  llm_request: 300 # Complex generation tasks
  tool_execution: 300 # Custom tool operations
permissions:
  file_read: true # Allow reading files
  file_write: true # Allow writing files
  git_commit: true # Allow git operations
  exec_bash: false # Allow bash execution (DANGEROUS!)
  github_api: false # Allow GitHub API (DANGEROUS!)
  allowed_bash_commands: [] # Pre-authorized bash commands
rag:
  embedding_model: "all-MiniLM-L6-v2"
  hybrid_alpha: 0.3 # BM25 weight (0-1)
  top_k: 4 # Number of results
  rerank_enabled: true
github: # 新
  token: ""
  owner: ""
  repo: ""
```
## Supported Languages
- **Python** (.py)
- **C++** (.cpp, .h, .cc)
- **C** (.c, .h)
- **JavaScript** (.js, .jsx)
- **TypeScript** (.ts, .tsx)
- **Java** (.java)
- **C#** (.cs)
- **Go** (.go)
## Examples
### Code Generation
```bash
python cli.py "Create a binary search function in Python"
python cli.py --lang cpp "Implement quicksort algorithm"
```
### Debugging
```bash
python cli.py "Debug @app.py - fix the IndexError"
python cli.py "Analyze @error.log and suggest fixes for @main.py"
```
### Documentation
```bash
python cli.py --mode doc "Summarize @README.md"
python cli.py /requirements # Generate requirements doc
```
### Testing
```bash
python cli.py "Generate unit tests for @calculator.py"
python cli.py "Run tests for @test_utils.py and fix failures"
```
### Design & Architecture
```bash
python cli.py /design # Generate architecture document
python cli.py "Design a REST API for user management"
```
### GitHub Integration # 新部分
```bash
python cli.py /create-pr "Add new feature" # Create PR with title
python cli.py /review-pr 5 # Review PR #5
python cli.py /commit-push-pr "Commit changes" # Commit, push, create PR
```
## RAG (Retrieval-Augmented Generation)
### How It Works
1. **Indexing**: Automatically scans and indexes your codebase
2. **Hybrid Search**: Combines semantic (embeddings) + keyword (BM25)
3. **Reranking**: Optional LLM-based result rerank
4. **Auto-refresh**: Updates index when files change (requires watchdog)
### Supported File Types
- Code: .py, .cpp, .c, .h, .js, .java, .cs, .go
- Docs: .md, .markdown, .txt, .text
### Optimization
- Uses tree-sitter for precise code chunking
- Caches embeddings locally
- Configurable chunk size and top-k results
## Timeouts Explained
| Timeout | Default | Use Case |
|---------|---------|----------|
| `bash_exec` | 300s | Long-running shell commands |
| `compile` | 300s | Large C++ projects |
| `unit_test` | 300s | Comprehensive test suites |
| `llm_request` | 300s | Complex generation tasks
| `tool_execution` | 300s | Custom tool operations |
**Tip:** Increase timeouts for large projects or slow networks.
## Permissions & Security
### Safe Defaults
- ✅ `file_read`: true
- ✅ `file_write`: true
- ✅ `git_commit`: true
- ❌ `exec_bash`: false (disabled by default)
- ❌ `github_api`: false (disabled by default)
### Enabling Bash Execution
**WARNING:** Only enable for trusted code!
```yaml
permissions:
  exec_bash: true # Use with caution!
```
### Pre-Authorizing Bash Commands
Add specific commands to `allowed_bash_commands`:
```yaml
permissions:
  allowed_bash_commands:
    - "git push"
    - "npm install"
```
### Enabling GitHub API # 新
**WARNING:** Only enable for trusted operations! Provide token in github section.
```yaml
permissions:
  github_api: true
```
## Troubleshooting
### "Index is empty"
- Run in a directory with code files
- Check RAG supported file extensions
- Verify files aren't in `.git/` or hidden directories
### "Timeout expired"
- Increase timeout in `config.yaml`
- Check network connection (for LLM requests)
- Optimize query complexity
### "Permission denied"
- Check permissions in `config.yaml`
- Verify file/directory access rights
### "LLM call failed"
- Verify `endpoint` is correct
- Check LLM service is running
- Validate `api_key` if required
## Advanced Usage
### Custom Tools
Add Python files to `.agent/tools/`:
```python
# .agent/tools/my_tool.py
def custom_analyzer(code):
    # Your analysis logic
    return results
```
Tools are auto-loaded and available to the agent.
### Context Management
```python
# View context
python cli.py /status
# Clear if context is too large
python cli.py /clear
```
### Headless Mode (JSON Output)
```bash
python cli.py --headless "List Python files" | jq .
```
## Best Practices
1. **Descriptive Prompts**: Be specific about what you want
2. **Use @ Mentions**: Reference exact files when debugging
3. **Check Status**: Monitor context size with `/status`
4. **Clear Context**: Use `/clear` if responses degrade
5. **Adjust Timeouts**: Increase for large operations
6. **Review Outputs**: Always verify generated code
7. **Version Control**: Commit before major agent operations
## Support & Resources
- Configuration: `config.yaml`
- Prompts: `prompts.yaml`
- Context: `AGENT.md`
- Tools: `.agent/tools/`
For more information, check the source code or configuration files.