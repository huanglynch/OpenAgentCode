import os
import sys
import click
import yaml
import json
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Agent
from context import ContextManager
from rag import RAGIndexer

DEFAULT_CONFIG = {
    'llm': {
        'provider': 'openai',
        'model': 'gpt-4',
        'api_key': 'os.environ/OPENAI_API_KEY',
        'base_url': 'https://api.openai.com/v1',
        'max_tokens': 4096,
        'temperature': 0.7
    },
    'paths': {
        'tools_dir': '.agent/tools',
        'context_file': 'AGENT.md',
        'help_file': 'help.md'
    },
    'workspace': './workspace',
    'permissions': {
        'allow_write': True,
        'allow_execute': True,
        'allow_network': False,
        'github_api': False,
        'allowed_bash_commands': []
    },
    'timeouts': {
        'compile': 300,
        'unit_test': 300,
        'tool_execution': 300
    },
    'github': {
        'token': '',
        'default_repo': ''
    },
    'rag': {
        'enabled': True,
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'top_k': 3
    }
}

DEFAULT_PROMPTS = {
    'system': '''You are an AI coding agent. You help users write, debug, and improve code.
You have access to tools for file operations, code execution, and testing.
Always think step by step and explain your reasoning.''',

    'planner': '''Given the user's request, create a detailed plan to accomplish the task.
Break down complex tasks into smaller, manageable steps.
Consider dependencies, error handling, and edge cases.''',

    'coder': '''Write clean, well-documented code following best practices.
Include error handling and input validation.
Add comments to explain complex logic.''',

    'tester': '''Create comprehensive tests for the code.
Include unit tests, edge cases, and error scenarios.
Ensure good test coverage.''',

    'debugger': '''Analyze the error or bug carefully.
Identify the root cause and propose a fix.
Explain why the error occurred and how the fix resolves it.'''
}


def load_config():
    """Load configuration from config.yaml"""
    original_cwd = os.getcwd()
    config_path = os.path.join(original_cwd, 'config.yaml')

    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        print("Run 'python cli.py --init' to initialize a new project.")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 处理工作目录
    workspace_dir = config.get('workspace', './workspace')
    if not os.path.isabs(workspace_dir):
        workspace_dir = os.path.join(original_cwd, workspace_dir)

    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)
        print(f"Created workspace directory: {workspace_dir}")

    # 切换到工作目录
    os.chdir(workspace_dir)
    print(f"Working directory: {os.getcwd()}")

    config['_original_cwd'] = original_cwd
    config['_current_workspace'] = workspace_dir

    # 处理 API key
    api_key = config['llm'].get('api_key', '')
    original_api_key = api_key

    if api_key.startswith('os.environ/'):
        env_var = api_key.split('/')[-1]
        config['llm']['api_key'] = os.environ.get(env_var, '')
    elif api_key.startswith('$') and api_key.endswith('$'):
        env_var = api_key[1:-1]
        config['llm']['api_key'] = os.environ.get(env_var, '')

    config['llm']['_original_api_key'] = original_api_key

    if not config['llm']['api_key']:
        print("Warning: API key not found in environment.")

    if 'permissions' in config and 'allowed_bash_commands' not in config['permissions']:
        config['permissions']['allowed_bash_commands'] = []

    return config


def load_prompts():
    """Load prompts from prompts.yaml"""
    if hasattr(load_config, '_original_cwd'):
        original_cwd = load_config._original_cwd
    else:
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == 'workspace' and os.path.exists(
                os.path.join(os.path.dirname(current_dir), 'prompts.yaml')):
            original_cwd = os.path.dirname(current_dir)
        else:
            original_cwd = current_dir

    prompts_path = os.path.join(original_cwd, 'prompts.yaml')

    if not os.path.exists(prompts_path):
        print(f"Warning: prompts.yaml not found at {prompts_path}, using defaults.")
        return DEFAULT_PROMPTS

    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def initialize_project():
    """Initialize a new OpenAgentCode project"""
    print("Initializing OpenAgentCode project...")

    original_cwd = os.getcwd()

    # 创建 .agent/tools 目录
    tools_dir = os.path.join(original_cwd, '.agent', 'tools')
    os.makedirs(tools_dir, exist_ok=True)
    print(f"Created directory: {tools_dir}")

    # 创建 config.yaml
    config_path = os.path.join(original_cwd, 'config.yaml')
    if os.path.exists(config_path):
        print(f"config.yaml already exists at {config_path}")
    else:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        print(f"Created config.yaml at {config_path}")

    # 创建 prompts.yaml
    prompts_path = os.path.join(original_cwd, 'prompts.yaml')
    if os.path.exists(prompts_path):
        print(f"prompts.yaml already exists at {prompts_path}")
    else:
        with open(prompts_path, 'w', encoding='utf-8') as f:
            yaml.dump(DEFAULT_PROMPTS, f, default_flow_style=False, sort_keys=False)
        print(f"Created prompts.yaml at {prompts_path}")

    # 创建 AGENT.md
    agent_md_path = os.path.join(original_cwd, 'AGENT.md')
    if os.path.exists(agent_md_path):
        print(f"AGENT.md already exists at {agent_md_path}")
    else:
        with open(agent_md_path, 'w', encoding='utf-8') as f:
            f.write('# Agent Context\n\noverview:\n\nhistory:\n\nfacts:\n\nlangs:\n')
        print(f"Created AGENT.md at {agent_md_path}")

    # 创建 workspace 目录
    workspace_dir = os.path.join(original_cwd, 'workspace')
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)
        print(f"Created workspace directory at {workspace_dir}")
    else:
        print(f"workspace directory already exists at {workspace_dir}")

    # 创建示例文件
    example_file = os.path.join(workspace_dir, 'example.py')
    if not os.path.exists(example_file):
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write('''def hello_world():
    """Example function"""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
''')
        print(f"Created example file at {example_file}")

    # 创建 .gitignore
    gitignore_path = os.path.join(original_cwd, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write('''# OpenAgentCode
workspace/
.agent/
*.pyc
__pycache__/
.env
.venv
venv/
*.log
.rag_index/
''')
        print(f"Created .gitignore at {gitignore_path}")

    # 创建 help.md
    create_help_file()

    # 创建 README.md
    readme_path = os.path.join(original_cwd, 'README.md')
    if not os.path.exists(readme_path):
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('''# OpenAgentCode Project

This is an OpenAgentCode project.

## Getting Started

1. Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Run the agent:
```bash
python cli.py
```

3. Or run a specific task:
```bash
python cli.py "Create a Python web scraper"
```

## Documentation

See `help.md` for detailed documentation.

## Configuration

Edit `config.yaml` to customize settings.
Edit `prompts.yaml` to customize agent prompts.
''')
        print(f"Created README.md at {readme_path}")

    print("\n✓ Project initialized successfully!")
    print("\nNext steps:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Edit config.yaml to customize settings")
    print("3. Run: python cli.py")


def create_help_file():
    """Create help.md with detailed documentation"""
    help_path = os.path.join(os.getcwd(), 'help.md')

    if os.path.exists(help_path):
        print(f"help.md already exists at {help_path}")
        return

    help_content = """# OpenAgentCode - Help Documentation

## Overview
OpenAgentCode is an AI-powered coding assistant that helps you write, debug, and improve code.

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize project
python cli.py --init

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run
python cli.py
```

## Modes
- **plan**: Create a detailed plan for the task
- **code**: Write code based on the plan or request
- **test**: Generate tests for the code
- **debug**: Debug and fix errors
- **auto**: Automatically choose the best mode (default)

## Commands

### Basic Usage
```bash
# Interactive mode
python cli.py

# Execute a single task
python cli.py "Create a Python web scraper"

# Specify mode
python cli.py --mode code "Write a function to sort a list"

# Headless output (JSON)
python cli.py --headless "Generate unit tests"

# Specify language
python cli.py --lang python "Create a REST API"

# Chat mode (no tool execution)
python cli.py --chat

# Initialize new project
python cli.py --init
```

### Slash Commands (Interactive Mode)

- `/help` - Show this help message
- `/clear` - Clear context and start fresh
- `/context` - Show current context
- `/files` - List files in workspace
- `/read <file>` - Read a file
- `/write <file>` - Write to a file
- `/ls` or `/dir` - List directory contents
- `/pwd` - Show current directory
- `/cd <dir>` - Change directory
- `/rag` - Rebuild RAG index
- `/config` - Show configuration
- `/modes` - List available modes
- `/quit` or `/exit` - Exit the program

### File References
Use `@filename` to reference files in your prompts:
```
"Add error handling to @main.py"
"Refactor @utils.py to use type hints"
"Compare @old_version.py with @new_version.py"
```

## Configuration

### config.yaml
Main configuration file:

```yaml
llm:
  provider: openai          # LLM provider
  model: gpt-4             # Model name
  api_key: os.environ/OPENAI_API_KEY
  base_url: https://api.openai.com/v1
  max_tokens: 4096
  temperature: 0.7

paths:
  tools_dir: .agent/tools
  context_file: AGENT.md
  help_file: help.md

workspace: ./workspace      # Working directory

permissions:
  allow_write: true
  allow_execute: true
  allow_network: false
  github_api: false
  allowed_bash_commands: []

timeouts:
  compile: 300
  unit_test: 300
  tool_execution: 300

github:
  token: ''
  default_repo: ''

rag:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 3
```

### prompts.yaml
Customize system prompts:

```yaml
system: "You are an AI coding agent..."
planner: "Create a detailed plan..."
coder: "Write clean code..."
tester: "Create comprehensive tests..."
debugger: "Analyze the error..."
```

## Features

### 1. RAG (Retrieval Augmented Generation)
Automatically indexes workspace files:
- Enabled by default
- Indexes all code files
- Retrieves relevant context
- Rebuild with `/rag` command

### 2. Context Management
Maintains conversation context:
- Task history
- Project overview
- Learned facts
- Programming languages

### 3. Tool System
Extensible tools for operations:
- File I/O (read, write, list)
- Code execution
- Testing
- Git operations
- GitHub API
- Custom tools

### 4. Multi-Mode Operation
Different modes for different tasks:
- **plan**: Strategic planning
- **code**: Code generation
- **test**: Test creation
- **debug**: Error fixing
- **auto**: Smart mode selection

## Security

### Permissions
Control what the agent can do:
- `allow_write`: File modifications
- `allow_execute`: Code execution
- `allow_network`: Network access
- `github_api`: GitHub operations
- `allowed_bash_commands`: Command whitelist

### Timeouts
Prevent runaway operations:
- Compilation timeout
- Test execution timeout
- Tool execution timeout

## Examples

### 1. Create a Web Application
```bash
python cli.py "Create a Flask web app with user authentication"
```

### 2. Debug Existing Code
```bash
python cli.py --mode debug "Fix the error in @app.py"
```

### 3. Generate Tests
```bash
python cli.py --mode test "Create unit tests for @calculator.py"
```

### 4. Interactive Development
```bash
python cli.py

>>> Create a REST API for a todo list
[Agent creates API...]

>>> Add database integration with SQLAlchemy
[Agent adds database...]

>>> Write tests for the API endpoints
[Agent writes tests...]

>>> /files
workspace/api.py
workspace/database.py
workspace/test_api.py

>>> /context
[Shows conversation context...]

>>> /quit
```

### 5. Refactoring
```bash
python cli.py "Refactor @legacy_code.py to use modern Python patterns"
```

### 6. Documentation
```bash
python cli.py "Add comprehensive docstrings to @module.py"
```

### 7. Code Review
```bash
python cli.py "Review @submission.py and suggest improvements"
```

## Advanced Usage

### Custom Tools
Add custom tools in `.agent/tools/`:

```python
# .agent/tools/my_tool.py

def analyze_performance(code: str) -> dict:
    '''
    Analyze code performance and suggest optimizations.

    Args:
        code: The code to analyze

    Returns:
        Performance analysis results
    '''
    # Implementation
    return {
        'time_complexity': 'O(n)',
        'space_complexity': 'O(1)',
        'suggestions': []
    }
```

### Environment Variables
Set in your shell or `.env`:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export ANTHROPIC_API_KEY="sk-ant-..."
export GITHUB_TOKEN="ghp_..."

# Custom endpoint
export CUSTOM_LLM_ENDPOINT="https://..."
```

### Custom LLM Provider
Edit `config.yaml`:

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229
  api_key: os.environ/ANTHROPIC_API_KEY
  base_url: https://api.anthropic.com
```

### GitHub Integration

1. Generate GitHub Personal Access Token
2. Set token:
```bash
export GITHUB_TOKEN="ghp_..."
```
3. Update config:
```yaml
github:
  token: os.environ/GITHUB_TOKEN
  default_repo: username/repo
permissions:
  github_api: true
```
4. Use GitHub commands:
```bash
python cli.py "Create a pull request for the new feature"
```

## Tips & Best Practices

1. **Be Specific**: Detailed prompts get better results
2. **Use References**: Use `@filename` to provide context
3. **Check Context**: Use `/context` to see what agent knows
4. **Clear When Needed**: Use `/clear` if agent gets confused
5. **Rebuild Index**: Use `/rag` after major code changes
6. **Review Code**: Always review generated code
7. **Appropriate Modes**: Use right mode for each task
8. **Version Control**: Use git for safety
9. **Test First**: Generate tests alongside code
10. **Iterative**: Break complex tasks into steps

## Troubleshooting

### API Key Issues
**Problem**: "API key not found"
**Solution**:
```bash
# Check environment
echo $OPENAI_API_KEY

# Set correctly
export OPENAI_API_KEY="sk-..."

# Verify config
cat config.yaml | grep api_key
```

### Permission Errors
**Problem**: "Permission denied for operation"
**Solution**: Edit `config.yaml`:
```yaml
permissions:
  allow_write: true
  allow_execute: true
```

### Context Too Large
**Problem**: "Token limit exceeded"
**Solutions**:
- Use `/clear` to reset
- Reduce `max_tokens` in config
- Be more specific in prompts

### RAG Not Finding Files
**Problem**: Agent doesn't see recent files
**Solutions**:
```bash
# Rebuild index
python cli.py
>>> /rag

# Check files
>>> /files

# Verify workspace
>>> /pwd
```

### Import Errors
**Problem**: "Module not found"
**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Slow Responses
**Problem**: Agent takes too long
**Solutions**:
- Use faster model in config
- Reduce `max_tokens`
- Disable RAG if not needed
- Clear context regularly

## FAQ

**Q: Can I use models other than OpenAI?**
A: Yes, configure any OpenAI-compatible API in `config.yaml`.

**Q: Is my code sent to the cloud?**
A: Yes, when using cloud LLM providers. Use local models for privacy.

**Q: Can I use this offline?**
A: Yes, with local LLM models (e.g., Ollama, LM Studio).

**Q: How do I add custom tools?**
A: Create Python files in `.agent/tools/` with documented functions.

**Q: Can I use multiple projects?**
A: Yes, each directory with `config.yaml` is a separate project.

**Q: How do I backup my work?**
A: Use git for version control, backup `workspace/` directory.

**Q: Can the agent access the internet?**
A: Only if `allow_network: true` in config (disabled by default).

**Q: How do I update OpenAgentCode?**
A: Pull latest code and reinstall: `pip install -e .`

## Support & Resources

- **Documentation**: This file (help.md)
- **GitHub**: [Repository URL]
- **Issues**: [Issues URL]
- **Examples**: Check `workspace/example.py`
- **Configuration**: `config.yaml` and `prompts.yaml`

## License

[Your License Here]

---

For more help, type `/help` in interactive mode or visit the documentation.
"""

    with open(help_path, 'w', encoding='utf-8') as f:
        f.write(help_content)

    print(f"Created help.md at {help_path}")


def show_help(config):
    """Display help information"""
    original_cwd = config.get('_original_cwd', '.')
    help_file = config.get('paths', {}).get('help_file', 'help.md')
    help_path = os.path.join(original_cwd, help_file)

    if os.path.exists(help_path):
        try:
            with open(help_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 只显示前 100 行，避免输出太长
                lines = content.split('\n')
                if len(lines) > 100:
                    print('\n'.join(lines[:100]))
                    print(f"\n... ({len(lines) - 100} more lines)")
                    print(f"\nRead full help: {help_path}")
                else:
                    print(content)
            return
        except Exception as e:
            print(f"Warning: Failed to read {help_path}: {e}")

    # 默认帮助
    print("OpenAgentCode - Quick Help")
    print("\nSlash Commands:")
    print("  /help     - Show detailed help")
    print("  /clear    - Clear context")
    print("  /context  - Show context")
    print("  /files    - List workspace files")
    print("  /read <file> - Read a file")
    print("  /write <file> - Write to a file")
    print("  /ls       - List directory")
    print("  /pwd      - Show current directory")
    print("  /cd <dir> - Change directory")
    print("  /rag      - Rebuild RAG index")
    print("  /config   - Show configuration")
    print("  /modes    - List available modes")
    print("  /quit     - Exit")
    print("\nModes: plan, code, test, debug, auto")
    print("Use @filename to reference files")
    print(f"\nFull documentation: {help_path}")


def format_markdown(text):
    """Format markdown text for terminal output"""
    # 简单的格式化，可以用 rich 库增强
    import re

    # 代码块
    text = re.sub(r'```(\w+)?\n(.*?)```', r'\n[\1 code]\n\2\n[/code]\n', text, flags=re.DOTALL)

    # 标题
    text = re.sub(r'^### (.+)$', r'\n▸ \1', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'\n▶ \1', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'\n━━━ \1 ━━━', text, flags=re.MULTILINE)

    return text


def resolve_at_mentions(prompt):
    """Resolve @filename references in the prompt"""
    import re

    # 找到所有 @filename 引用
    pattern = r'@([\w\./\\-]+\.\w+)'
    matches = re.findall(pattern, prompt)

    resolved = prompt
    for filename in matches:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                replacement = f"\n--- Content of {filename} ---\n{content}\n--- End of {filename} ---\n"
                resolved = resolved.replace(f'@{filename}', replacement)
            except Exception as e:
                resolved = resolved.replace(f'@{filename}', f"[Error reading {filename}: {e}]")
        else:
            resolved = resolved.replace(f'@{filename}', f"[File not found: {filename}]")

    return resolved


def handle_slash_command(agent, command, mode, lang, config):
    """Handle slash commands"""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ''

    if cmd == 'help':
        show_help(config)

    elif cmd == 'clear':
        agent.context_manager.clear()
        print("✓ Context cleared.")

    elif cmd == 'context':
        context = agent.context_manager.load()
        print(yaml.dump(context, default_flow_style=False))

    elif cmd == 'files':
        print("Files in workspace:")
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if not file.startswith('.'):
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    print(f"  {filepath} ({size} bytes)")

    elif cmd == 'read':
        if not args:
            print("Usage: /read <filename>")
            return

        if os.path.exists(args):
            try:
                with open(args, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"\n━━━ {args} ━━━")
                    print(content)
                    print(f"━━━ End of {args} ━━━\n")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"File not found: {args}")

    elif cmd == 'write':
        if not args:
            print("Usage: /write <filename>")
            print("Enter content (end with Ctrl+D on Unix or Ctrl+Z on Windows):")
            return

        print(f"Enter content for {args} (end with Ctrl+D or Ctrl+Z):")
        try:
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break

            content = '\n'.join(lines)
            with open(args, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Written to {args}")
        except Exception as e:
            print(f"Error writing file: {e}")

    elif cmd in ['ls', 'dir']:
        print("Directory contents:")
        try:
            for item in os.listdir('.'):
                if os.path.isdir(item):
                    print(f"  [DIR]  {item}/")
                else:
                    size = os.path.getsize(item)
                    print(f"  [FILE] {item} ({size} bytes)")
        except Exception as e:
            print(f"Error listing directory: {e}")

    elif cmd == 'pwd':
        print(f"Current directory: {os.getcwd()}")

    elif cmd == 'cd':
        if not args:
            print("Usage: /cd <directory>")
            return

        try:
            os.chdir(args)
            print(f"✓ Changed to: {os.getcwd()}")
        except Exception as e:
            print(f"Error changing directory: {e}")

    elif cmd == 'rag':
        print("Rebuilding RAG index...")
        try:
            workspace = config.get('_current_workspace', '.')
            indexer = RAGIndexer(config, workspace)
            print("✓ RAG index rebuilt.")
        except Exception as e:
            print(f"Error rebuilding RAG index: {e}")

    elif cmd == 'config':
        print("Current configuration:")
        # 不显示 API key
        safe_config = config.copy()
        if 'llm' in safe_config and 'api_key' in safe_config['llm']:
            api_key = safe_config['llm']['api_key']
            if api_key:
                safe_config['llm']['api_key'] = api_key[:8] + '...' if len(api_key) > 8 else '***'
        print(yaml.dump(safe_config, default_flow_style=False))

    elif cmd == 'modes':
        print("Available modes:")
        print("  plan   - Create a detailed plan")
        print("  code   - Write code")
        print("  test   - Generate tests")
        print("  debug  - Debug and fix errors")
        print("  auto   - Automatically choose mode (default)")
        print(f"\nCurrent mode: {mode}")

    elif cmd in ['quit', 'exit']:
        print("Goodbye!")
        sys.exit(0)

    else:
        print(f"Unknown command: /{cmd}")
        print("Type /help for available commands")


def interactive_mode(agent, mode, headless, lang, config):
    """Interactive REPL mode"""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OpenAgentCode Interactive Mode")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Type /help for commands, /quit to exit")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()

            if not prompt:
                continue

            # Handle slash commands
            if prompt.startswith('/'):
                handle_slash_command(agent, prompt[1:], mode, lang, config)
                continue

            # Process regular prompt
            resolved_prompt = resolve_at_mentions(prompt)

            print("\n[Processing...]")
            result = agent.infer(resolved_prompt, mode=mode, lang=lang)

            if headless:
                print(json.dumps(result, indent=2))
            else:
                print("\n" + format_markdown(result))
            print()

        except KeyboardInterrupt:
            print("\n[Interrupted. Type /quit to exit]")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def interactive_chat(config):
    """Simple chat interface without tool execution"""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OpenAgentCode Chat Mode")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Chat Mode: Converse with AI directly.")
    print("Commands: /exit, /quit, /help, /config, /status, /model, /clear, /rag <query>")
    print()

    from llm import LLMClient

    llm = LLMClient(config['llm'])
    conversation = []

    prompts = load_prompts()
    system_msg = prompts.get('system', DEFAULT_PROMPTS['system'])

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1] if len(cmd_parts) > 1 else ''

                if cmd in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                elif cmd == 'help':
                    print("\n=== Chat Mode Commands ===")
                    print("/exit, /quit     - Exit chat mode")
                    print("/help            - Show this help message")
                    print("/config          - Show current configuration")
                    print("/status          - Show chat status")
                    print("/model           - Show current model info")
                    print("/clear           - Clear conversation history")
                    print("/rag <query>     - Search RAG index (if enabled)")
                    print("\nType your message to chat with AI.\n")
                    continue

                elif cmd == 'config':
                    print("\n=== Current Configuration ===")
                    # Don't show API key
                    safe_config = config.copy()
                    if 'llm' in safe_config and 'api_key' in safe_config['llm']:
                        api_key = safe_config['llm']['api_key']
                        if api_key:
                            safe_config['llm']['api_key'] = api_key[:8] + '...' if len(api_key) > 8 else '***'
                    print(yaml.dump(safe_config, default_flow_style=False))
                    continue

                elif cmd == 'status':
                    print("\n=== Chat Status ===")
                    print(f"Model: {config['llm']['model']}")
                    print(f"Endpoint: {config['llm']['endpoint']}")
                    print(f"Conversation turns: {len(conversation) // 2}")
                    print(f"Max tokens: {config['llm']['max_tokens']}")
                    print(f"Temperature: {config['llm']['temperature']}\n")
                    continue

                elif cmd == 'model':
                    print("\n=== Model Information ===")
                    print(f"Provider: {config['llm'].get('provider', 'unknown')}")
                    print(f"Model: {config['llm']['model']}")
                    print(f"Endpoint: {config['llm']['endpoint']}")
                    print(f"Max tokens: {config['llm']['max_tokens']}")
                    print(f"Temperature: {config['llm']['temperature']}\n")
                    continue

                elif cmd == 'clear':
                    conversation.clear()
                    print("✓ Conversation history cleared.\n")
                    continue

                elif cmd == 'rag':
                    if not args:
                        print("Usage: /rag <query>")
                        print("Example: /rag how to use async in python\n")
                        continue

                    if not config.get('rag', {}).get('enabled', False):
                        print("RAG is not enabled in config.yaml\n")
                        continue

                    try:
                        from rag_vector import VectorRAG
                        rag = VectorRAG(config)
                        results = rag.search(args, config['rag']['top_k'], 'code', None)

                        if results:
                            print(f"\n=== RAG Search Results for: {args} ===")
                            for i, path in enumerate(results, 1):
                                print(f"{i}. {path}")
                            print()
                        else:
                            print(f"No results found for: {args}\n")
                    except Exception as e:
                        print(f"RAG search failed: {e}\n")
                    continue

                else:
                    print(f"Unknown command: /{cmd}")
                    print("Type /help for available commands\n")
                    continue

            # Regular chat message
            conversation.append({"role": "user", "content": user_input})

            messages = [{"role": "system", "content": system_msg}] + conversation

            print("\n[Thinking...]")
            response = llm.chat(messages)

            conversation.append({"role": "assistant", "content": response})

            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n[Interrupted. Type /quit to exit]")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()

@click.command()
@click.argument('prompt', required=False, default='')
@click.option('--mode', '-m', type=click.Choice(['plan', 'code', 'test', 'debug', 'auto']), default='auto',
              help='Agent mode')
@click.option('--headless', is_flag=True, help='Output JSON instead of formatted text')
@click.option('--lang', '-l', default='', help='Programming language')
@click.option('--chat', is_flag=True, help='Enter chat mode (no tool execution)')
@click.option('--init', is_flag=True, help='Initialize a new project')
def main(prompt, mode, headless, lang, chat, init):
    """
    OpenAgentCode - AI Coding Agent
    Examples:
        python cli.py --init
        python cli.py
        python cli.py "Create a web scraper"
        python cli.py --mode code "Write a function"
        python cli.py --chat
    """
    # Handle initialization
    if init:
        initialize_project()
        return
    # Load configuration
    original_cwd = os.getcwd()
    config_path = os.path.join(original_cwd, 'config.yaml')
    if not os.path.exists(config_path):
        print("No config.yaml found. Initializing project...")
        initialize_project()
        print("\nPlease set your API key and run again:")
        print(" export OPENAI_API_KEY='your-key-here'")
        print(" python cli.py")
        return
    # Load config and prompts
    config = load_config()
    load_config._original_cwd = config['_original_cwd']
    prompts = load_prompts()
    # Handle chat mode (no agent needed)
    if chat:
        interactive_chat(config)
        return
    # Create agent (only for non-chat modes)
    try:
        context_manager = ContextManager(config)
        agent = Agent(config, prompts, context_manager)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return
    # Handle interactive mode (no prompt)
    if not prompt:
        interactive_mode(agent, mode, headless, lang, config)
        return


if __name__ == '__main__':
    main()