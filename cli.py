# cli.py
import click
import base64
import os
import sys
import json
import yaml
import difflib  # 添加此行
import requests
from agent import Agent
from context import ContextManager

DEFAULT_CONFIG = {
    'llm': {
        'endpoint': "http://localhost:8000/api/chat",
        'model': "AI:Pro",
        'temperature': 0.7,
        'max_tokens': 8192,
        'api_key': ""
    },
    'paths': {
        'prompts_file': "prompts.yaml",
        'context_file': "AGENT.md",
        'tools_dir': ".agent/tools/",
        'embed_cache_dir': "cached_models/",  # 或 "./cached_models/"
        'help_file': "help.md"  # 添加 help 文件路径
    },
    'permissions': {
        'file_read': True,
        'file_write': True,
        'git_commit': True,
        'exec_bash': False,
        'github_api': False,  # 新权限
        'allowed_bash_commands': []  # 新增：预授权的bash命令列表
    },
    'modes': {
        'default': "code"
    },
    'timeouts': {
        'bash_exec': 300,
        'compile': 300,
        'unit_test': 300,
        'llm_request': 300,
        'tool_execution': 300
    },
    'rag': {
        'embedding_model': "all-MiniLM-L6-v2",
        'hybrid_alpha': 0.3,
        'top_k': 4,
        'index_refresh_interval': 300,
        'chunk_size': 512,
        'rerank_enabled': True
    },
    'languages': {
        'supported': ["python", "cpp", "c", "js", "java", "cs", "go"],
        'default': "python"
    },
    'tasks_optimizations': {
        'debug': {
            'chunk_method': "function",
            'rerank_prompt': "Rank snippets by relevance to bug: {error_desc}"
        },
        'ut': {
            'query_expand': "Retrieve similar tests and coverage for {func} in {lang}",
            'post_validate': True
        },
        'doc': {
            'chunk_method': "section",
            'rerank_prompt': "Rank by redundancy and clarity for doc optimization"
        },
        'requirements': {
            'template': "Generate requirements.md: functional, non-functional, constraints."
        },
        'design': {
            'template': "Design architecture: UML text, components, in {lang}."
        },
        'optimize': {
            'template': "Optimize code/doc: performance, readability, refactor suggestions."
        }
    },
    'github': {  # 新配置
        'token': "",
        'owner': "",
        'repo': ""
    }
}

@click.command()
@click.argument('prompt', required=False)
@click.option('--mode', default='code', help='Mode: code or doc')
@click.option('--headless', is_flag=True, help='Output JSON only')
@click.option('--lang', default=None, help='Language: python, cpp, etc.')
@click.option('--chat', is_flag=True, help='Enter chat mode for direct AI conversation')
def main(prompt, mode, headless, lang, chat):
    """OpenAgentCode CLI entry point."""
    if not os.path.exists('config.yaml'):
        initialize_project()
    config = load_config()
    prompts = load_prompts()
    context_manager = ContextManager(config)
    agent = Agent(config, prompts, context_manager)
    if chat:
        interactive_chat(config)
        return
    if not prompt:
        interactive_mode(agent, mode, headless, lang, config)  # 添加 config
        return
    if prompt.startswith('/'):
        handle_slash_command(agent, prompt[1:], mode, lang, config)
        return
    resolved_prompt = resolve_at_mentions(prompt)
    result = agent.infer(resolved_prompt, mode=mode, lang=lang)
    if headless:
        print(json.dumps(result))
    else:
        print(format_markdown(result))

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    api_key = config['llm'].get('api_key', '')
    original_api_key = api_key  # 保存原始值
    if api_key.startswith('os.environ/'):
        env_var = api_key.split('/')[-1]
        config['llm']['api_key'] = os.environ.get(env_var, '')
    elif api_key.startswith('$') and api_key.endswith('$'):  # 新支持 $VAR$ 格式
        env_var = api_key[1:-1]
        config['llm']['api_key'] = os.environ.get(env_var, '')
    config['llm']['_original_api_key'] = original_api_key  # 存储原始占位符或值
    # 新增警告：如果 api_key 为空，打印警告
    if not config['llm']['api_key']:
        print("Warning: API key not found in environment.")
    if original_api_key and (original_api_key.startswith('$') or original_api_key.startswith('os.environ/')) and not \
       config['llm']['api_key']:
        print(f"Warning: Environment variable for api_key ({env_var}) not found.")
    # 新增：兼容allowed_bash_commands
    if 'permissions' in config and 'allowed_bash_commands' not in config['permissions']:
        config['permissions']['allowed_bash_commands'] = []
    return config

def load_prompts():
    with open('prompts.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_at_mentions(prompt):
    parts = prompt.split()
    resolved = []
    for part in parts:
        if part.startswith('@'):
            filename = part[1:]
            matched_file = fuzzy_match_file(filename)
            if matched_file:
                try:
                    with open(matched_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    resolved.append(f"{part}:\n{content}")
                except Exception as e:
                    print(f"Warning: Failed to read {matched_file}: {e}")
                    resolved.append(part)
            else:
                resolved.append(part)
        else:
            resolved.append(part)
    return ' '.join(resolved)

def fuzzy_match_file(filename):
    all_files = []
    for root, _, files in os.walk('.'):
        # 使用 os.path.normpath 规范化路径
        root_norm = os.path.normpath(root)
        # 跨平台检查隐藏目录
        skip_dirs = ['.git', '.agent', 'cached_models', '__pycache__']
        if any(skip_dir in root_norm.split(os.path.sep) for skip_dir in skip_dirs):
            continue
        for file in files:
            all_files.append(os.path.join(root, file))
    basenames = [os.path.basename(f) for f in all_files]
    matches = difflib.get_close_matches(filename, basenames, n=1, cutoff=0.6)
    if matches:
        for f in all_files:
            if os.path.basename(f) == matches[0]:
                return f
    return None

def interactive_mode(agent, mode, headless, lang, config):  # 添加 config 参数
    print("OpenAgentCode Interactive Mode")
    print("Commands: /help, /exit, /clear")
    print("Mention files with @filename")
    print()
    while True:
        try:
            user_input = input(">> ")
            if not user_input.strip():
                continue
            if user_input.lower() in ['/exit', '/quit']:
                print("Goodbye!")
                break
            if user_input.startswith('/'):
                handle_slash_command(agent, user_input[1:], mode, lang, config)  # 已传入 config
            else:
                resolved = resolve_at_mentions(user_input)
                result = agent.infer(resolved, mode=mode, lang=lang)
                if headless:
                    print(json.dumps(result))
                else:
                    print(format_markdown(result))
        except KeyboardInterrupt:
            print("\nUse /exit to quit")
        except Exception as e:
            print(f"Error: {e}")

def interactive_chat(config):
    print("Chat Mode: Converse with AI directly. Commands: /exit, /quit, /help, /config, /status, /model, /clear, /rag <query>")
    history = []
    while True:
        try:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ['/exit', '/quit']:
                print("Goodbye!")
                break
            if user_input.startswith('/'):
                command = user_input[1:].strip()
                cmd_lower = command.lower()
                if cmd_lower.startswith('model'):
                    parts = command.split()
                    if len(parts) > 1:
                        model_str = parts[1]
                        if ',' in model_str:
                            endpoint_type, model_name = model_str.split(',', 1)
                            endpoint_type = endpoint_type.lower()
                            endpoint_key = f'endpoint_{endpoint_type}'
                            if endpoint_key in config['llm']:
                                config['llm']['endpoint'] = config['llm'][endpoint_key]
                                config['llm']['model'] = model_name
                                print(f"Switched to model {model_name} with {endpoint_type} endpoint: {config['llm']['endpoint']}")
                                # 保存配置变更到文件，确保 api_key 不写入实际值
                                temp_api_key = config['llm']['api_key']
                                config['llm']['api_key'] = config['llm']['_original_api_key']
                                print(yaml.dump(config, default_flow_style=False))
                                with open('config.yaml', 'w', encoding='utf-8') as f:
                                    yaml.dump(config, f, default_flow_style=False)
                                config['llm']['api_key'] = temp_api_key # 恢复内存
                            else:
                                print(f"Endpoint for {endpoint_type} not defined in config.")
                        else:
                            print("Invalid format. Use: /model endpoint_type,model_name (e.g., /model vllm,AI:Pro)")
                    else:
                        print("Usage: /model [endpoint_type,model_name]")
                        print(f"Available models: {', '.join(config.get('models', []))}")
                    continue
                elif cmd_lower.startswith('rag'):
                    query = command[3:].strip()
                    if not query:
                        print("Usage: /rag <query>")
                        continue
                    # 临时创建 ContextManager 和 VectorRAG
                    from context import ContextManager
                    from rag_vector import VectorRAG
                    context_manager = ContextManager(config)
                    rag = VectorRAG(config)
                    # 执行搜索
                    rag_results = rag.search(query, config['rag']['top_k'], config['modes']['default'], config['languages']['default'])
                    # 注入 RAG 结果（复制 agent.inject_rag_results 逻辑）
                    injected = []
                    for path in rag_results:
                        try:
                            if os.path.getsize(path) < 5000:
                                encodings = ['utf-8', 'gbk', 'latin-1']
                                content = None
                                for encoding in encodings:
                                    try:
                                        with open(path, 'r', encoding=encoding) as f:
                                            content = f.read()
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                if content is None:
                                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()
                            else:
                                encodings = ['utf-8', 'gbk', 'latin-1']
                                content = None
                                for encoding in encodings:
                                    try:
                                        with open(path, 'r', encoding=encoding) as f:
                                            content = f.read()
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                if content is None:
                                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()
                                content = context_manager.compress(content)
                            injected.append(f"File: {path}\n{content}")
                        except Exception as e:
                            print(f"Warning: Failed to inject {path}: {e}")
                            continue
                    injected_rag = '\n'.join(injected)
                    # 构建 messages with RAG context
                    messages = [{'role': 'system', 'content': 'You are a helpful assistant. Use the following context to answer the query:\n' + injected_rag}] + history + [{'role': 'user', 'content': query}]
                    # 设置 model 和 endpoint (非视觉)
                    model = config['llm']['model']
                    endpoint = config['llm']['endpoint']
                    # 继续到 payload 和请求部分
                    history.append({'role': 'user', 'content': query})
                    payload = {
                        'model': model,
                        'messages': messages,
                        'temperature': config['llm']['temperature'],
                        'max_tokens': config['llm']['max_tokens'],
                        'stream': True
                    }
                    headers = {}
                    if config['llm'].get('api_key'):
                        headers['Authorization'] = f'Bearer {config["llm"]["api_key"]}'
                    try:
                        response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                        response.raise_for_status()
                        print("AI: ", end='', flush=True)
                        content = ''
                        for line in response.iter_lines():
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:]
                                try:
                                    chunk = json.loads(line_str)
                                    delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                    if delta:
                                        content += delta
                                        print(delta, end='', flush=True)
                                except json.JSONDecodeError:
                                    continue
                        print()
                    except Exception as stream_error:
                        print(f"Stream mode failed: {stream_error}. Falling back to non-stream mode.")
                        payload['stream'] = False
                        response = requests.post(endpoint, json=payload, headers=headers)
                        response.raise_for_status()
                        content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                        print("AI:", content)
                    history.append({'role': 'assistant', 'content': content})
                    continue
                else:
                    # 临时创建 agent
                    from agent import Agent
                    from context import ContextManager
                    prompts = load_prompts()
                    context_manager = ContextManager(config)
                    agent = Agent(config, prompts, context_manager)
                    handle_slash_command(agent, command, mode='code', lang=None, config=config)
                    continue
            # 新增: 检查图片输入
            image_url = None
            image_base64 = None
            text_content = user_input
            # 检查是否包含 URL
            if 'http' in user_input and any(user_input.lower().endswith(ext) for ext in ['.jpg', '.png', '.gif', '.jpeg']):
                parts = user_input.rsplit('http', 1)
                if len(parts) == 2:
                    text_content = parts[0].strip()
                    image_url = 'http' + parts[1].strip()
            # 否则检查本地路径
            elif any(user_input.lower().endswith(ext) for ext in ['.jpg', '.png', '.gif', '.jpeg']):
                parts = user_input.rsplit(' ', 1) # 假设 "Describe: /path/to/image.jpg"
                if len(parts) == 2 and os.path.exists(parts[1].strip()):
                    local_path = parts[1].strip()
                    text_content = parts[0].strip()
                    try:
                        with open(local_path, "rb") as image_file:
                            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                        mime_type = 'image/jpeg' if local_path.lower().endswith('.jpg') or local_path.lower().endswith('.jpeg') else 'image/png'
                    except Exception as e:
                        print(f"Error reading local image: {e}")
                        continue
            # 构建 messages
            if image_url or image_base64:
                model = config['llm'].get('vision_model', config['llm']['model'])
                endpoint = config['llm'].get('endpoint_vision', config['llm']['endpoint'])
                if image_url:
                    image_content = {'type': 'image_url', 'image_url': {'url': image_url}}
                else:
                    image_content = {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_base64}'}}
                messages = [{'role': 'system', 'content': 'You are a helpful assistant with vision capabilities.'}] + history + [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': text_content},
                        image_content
                    ]
                }]
            else:
                model = config['llm']['model']
                endpoint = config['llm']['endpoint']
                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + history + [{'role': 'user', 'content': user_input}]
            history.append({'role': 'user', 'content': user_input})
            payload = {
                'model': model,
                'messages': messages,
                'temperature': config['llm']['temperature'],
                'max_tokens': config['llm']['max_tokens'],
                'stream': True
            }
            headers = {}
            if config['llm'].get('api_key'):
                headers['Authorization'] = f'Bearer {config["llm"]["api_key"]}'
            try:
                response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                response.raise_for_status()
                print("AI: ", end='', flush=True)
                content = ''
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                        try:
                            chunk = json.loads(line_str)
                            delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if delta:
                                content += delta
                                print(delta, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
                print()
            except Exception as stream_error:
                print(f"Stream mode failed: {stream_error}. Falling back to non-stream mode.")
                payload['stream'] = False
                response = requests.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                print("AI:", content)
            history.append({'role': 'assistant', 'content': content})
        except KeyboardInterrupt:
            print("\nUse /exit to quit")
        except Exception as e:
            print(f"Error: {e}")

def handle_slash_command(agent, command, mode, lang, config):
    """Handle slash commands"""
    cmd_lower = command.lower().strip()
    # Help command
    if cmd_lower == 'help':
        show_help(config)
        return
    # Clear context
    if cmd_lower == 'clear':
        agent.context_manager.clear()
        print("Context cleared.")
        return
    # Chat command
    if cmd_lower == 'chat':
        interactive_chat(config)
        return
    # Status command
    if cmd_lower == 'status':
        context = agent.context_manager.load()
        print(f"Context overview: {context.get('overview', 'N/A')[:100]}...")
        print(f"History entries: {len(context.get('history', []))}")
        print(f"Facts: {len(context.get('facts', []))}")
        print(f"Languages: {', '.join(context.get('langs', []))}")
        return
    # Config command
    if cmd_lower == 'config':
        temp_api_key = config['llm']['api_key']
        config['llm']['api_key'] = config['llm']['_original_api_key']
        print(yaml.dump(config, default_flow_style=False))
        config['llm']['api_key'] = temp_api_key  # 恢复内存
        return
    # 新添加这里
    if cmd_lower == 'reload':
        config = load_config()  # 重新加载 config.yaml
        prompts = load_prompts()  # 重新加载 prompts.yaml
        agent.config = config  # 更新 agent 的 config
        agent.prompts = prompts  # 更新 agent 的 prompts
        print("Configuration and prompts reloaded from files.")
        return
    # 新命令：create-pr
    if cmd_lower.startswith('create-pr'):
        task = f"/create-pr {command[9:].strip()}" if len(command) > 9 else "/create-pr"
        result = agent.infer(task, mode=mode, lang=lang)
        print(format_markdown(result))
        return
    # 新命令：review-pr
    if cmd_lower.startswith('review-pr'):
        task = f"/review-pr {command[9:].strip()}" if len(command) > 9 else "/review-pr"
        result = agent.infer(task, mode=mode, lang=lang)
        print(format_markdown(result))
        return
    # 新命令：/model
    if cmd_lower.startswith('model'):
        parts = command.split()
        if len(parts) > 1:
            model_str = parts[1]
            if ',' in model_str:
                endpoint_type, model_name = model_str.split(',', 1)
                endpoint_type = endpoint_type.lower()
                endpoint_key = f'endpoint_{endpoint_type}'
                if endpoint_key in config['llm']:
                    config['llm']['endpoint'] = config['llm'][endpoint_key]
                    config['llm']['model'] = model_name
                    print(f"Switched to model {model_name} with {endpoint_type} endpoint: {config['llm']['endpoint']}")
                    # 保存配置变更到文件，确保 api_key 不写入实际值
                    temp_api_key = config['llm']['api_key']
                    config['llm']['api_key'] = config['llm']['_original_api_key']
                    print(yaml.dump(config, default_flow_style=False))
                    with open('config.yaml', 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    config['llm']['api_key'] = temp_api_key  # 恢复内存
                else:
                    print(f"Endpoint for {endpoint_type} not defined in config.")
            else:
                print("Invalid format. Use: /model endpoint_type,model_name (e.g., /model vllm,AI:Pro)")
        else:
            print("Usage: /model [endpoint_type,model_name]")
            print(f"Available models: {', '.join(config.get('models', []))}")
        return
    # 新增：permissions
    if cmd_lower.startswith('permissions'):
        args = command[11:].strip()
        if args:
            if args not in config['permissions']['allowed_bash_commands']:
                config['permissions']['allowed_bash_commands'].append(args)
            # 保存配置
            temp_api_key = config['llm']['api_key']
            config['llm']['api_key'] = config['llm']['_original_api_key']
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            config['llm']['api_key'] = temp_api_key
            print(f"Added allowed bash command: {args}")
        else:
            if config['permissions']['exec_bash']:
                print("exec_bash enabled: All bash commands allowed.")
            else:
                allowed = config['permissions']['allowed_bash_commands'] or ['None']
                print("Allowed bash commands:\n" + '\n'.join(allowed))
        return
    # 新增：commit-push-pr
    if cmd_lower.startswith('commit-push-pr'):
        task = f"/commit-push-pr {command[14:].strip()}" if len(command) > 14 else "/commit-push-pr"
        result = agent.infer(task, mode=mode, lang=lang)
        print(format_markdown(result))
        return
    # Execute as task
    task = f"/{command}"
    result = agent.infer(task, mode=mode, lang=lang)
    print(format_markdown(result))

def show_help(config):
    """Display help information"""
    help_file = config.get('paths', {}).get('help_file', 'help.md')
    # Try to load help.md
    if os.path.exists(help_file):
        try:
            with open(help_file, 'r', encoding='utf-8') as f:
                print(f.read())
            return
        except Exception as e:
            print(f"Warning: Failed to read {help_file}: {e}")

def initialize_project():
    """Initialize a new OpenAgentCode project"""
    print("Initializing OpenAgentCode project...")
    os.makedirs('.agent/tools', exist_ok=True)
    # Create config.yaml
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
    # Create prompts.yaml with complete templates
    DEFAULT_PROMPTS = {
        'base_prompt': 'You are OpenAgentCode, AI agent for {mode} tasks in {lang}.\nStep-by-step: explore, plan, execute, verify.\nContext: {context}',
        'summarize': 'Compress to key points (< {token_limit} tokens): {content}',
        'code_explore': 'Analyze repo: structure, deps in {lang}. Query: {query}. Use RAG snippets for semantic insights.',
        'code_plan': 'Plan for: {query}. Steps, changes in {lang}. Consider hybrid RAG dependencies.',
        'code_execute': 'Implement: {task} in {lang}. Code only. Verify with edges from RAG.',
        'doc_summarize': 'Summarize doc: keys, ideas. Query: {query}. Optimize clarity via reranked sections.',
        'doc_optimize': 'Optimize: reduce redundancy, clarify. Input: {content}. Use chunked insights.',
        'error_handle': 'Error: {error}.\nStep 1: Identify symptoms and error type.\nStep 2: Trace potential root cause in the code or dependencies.\nStep 3: Suggest targeted fixes with code snippets, prioritized by likelihood.\nStep 4: Verify suggestions against potential side effects.\nRerank fixes by relevance.\nOutput in JSON format: {"root_cause": "description", "fixes": [{"fix_id": 1, "description": "desc", "code_snippet": "code"}, ...]}',
        'debug_rerank': 'Rank code snippets by bug-fixing relevance: {error_desc}. Top {k}. Lang: {lang}.',
        'ut_expand': 'Expand query for UT: similar tests, edges in {lang}. Original: {query}.',
        'doc_rerank': 'Rank doc sections by optimization potential: redundancy, clarity. Top {k}.',
        'requirements_gen': 'Generate requirements doc: {query}. Include functional/non-functional, constraints.',
        'design_gen': 'Design software: {query} in {lang}. Output text UML, architecture.',
        'optimize_task': 'Optimize {type}: {query}. Suggestions for performance/readability in {lang}.',
        'create_pr': 'Plan to create PR: {query}. Generate title, body based on changes. Use RAG for diff analysis.',  # 新
        'review_pr': 'Review PR: {query}. Analyze diff for bugs, security, style in {lang}. Output comment with suggestions.\nStep 1: Identify changes.\nStep 2: Check functionality, quality, deps.\nStep 3: Suggest fixes.\nOutput JSON: {"comment": "full review text"}'  # 新
    }
    with open('prompts.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(DEFAULT_PROMPTS, f, default_flow_style=False)
    # Create AGENT.md
    with open('AGENT.md', 'w', encoding='utf-8') as f:
        f.write('# Agent Context\noverview:\nhistory:\nfacts:\nlangs:\n')
    # Create help.md
    create_help_file()
    print("Project initialized! Edit config.yaml to customize.")

def create_help_file():
    """Create help.md with detailed documentation"""
    if os.path.exists('help.md'):
        print("help.md already exists, skipping creation.")
        return
    help_content = """# OpenAgentCode - Help Documentation
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
"""
    with open('help.md', 'w', encoding='utf-8') as f:
        f.write(help_content)

def format_markdown(result):
    """Format result as markdown"""
    return f"**Plan:** {result.get('plan', '')}\n\n**Output:**\n{result.get('output', '')}"

if __name__ == '__main__':
    main()