# agent.py (AI 代理核心) - Full implementation with NVIDIA thinking model support

import requests
import json
from rag_vector import VectorRAG
from context import ContextManager
import multiprocessing
import os
import yaml


class Agent:
    def __init__(self, config, prompts, context_manager):
        self.config = config
        self.prompts = prompts
        self.context_manager = context_manager
        self.rag = VectorRAG(config)

    def infer(self, task, mode='code', lang=None):
        if not lang:
            lang = self.detect_language(task)

        context = self.context_manager.load()

        # 检查上下文大小，避免过度压缩触发额外LLM调用
        context_tokens = self.context_manager.token_monitor(yaml.dump(context))
        if context_tokens > self.config['llm']['max_tokens'] * 0.5:
            print(f"Debug: Compressing context ({context_tokens} tokens)")
            context = self.context_manager.compress(yaml.dump(context))

        rag_results = self.rag.search(task, self.config['rag']['top_k'], mode, lang)
        injected_rag = self.inject_rag_results(rag_results)
        prompt = self.build_prompt(task, mode, lang, context, injected_rag)

        # 自动优化 prompt，如果是 debug 相关
        if 'debug' in task.lower() or 'error' in task.lower():
            print("Debug: Optimizing prompt for debug task")
            prompt = self.optimize_prompt(prompt)

        print(f"Debug: Calling LLM with prompt length: {len(prompt)}")
        response = self.call_llm(prompt)

        if not response or response.startswith("Error:"):
            return {'plan': 'LLM call failed', 'output': response}

        parsed = self.parse_output(response)

        from executor import get_executor
        executor = get_executor(mode, self.config)

        results = []
        for action in parsed.get('actions', []):
            try:
                result = executor.execute(action, lang=lang)
                results.append(result)
            except Exception as e:
                results.append(f"Action execution failed: {e}")

        update_data = {
            'task': task,
            'plan': parsed.get('plan', ''),
            'results': '\n'.join(results)
        }
        self.context_manager.update(update_data)

        return {
            'plan': parsed.get('plan', ''),
            'output': '\n'.join(results)
        }

    def build_prompt(self, task, mode, lang, context, injected_rag):
        base = self.prompts.get('base_prompt', '').format(
            mode=mode,
            lang=lang,
            context=yaml.dump(context) + "\n" + injected_rag
        )

        if 'debug' in task.lower() or 'error' in task.lower():
            base += "\n" + self.prompts.get('error_handle', '').format(error=task, lang=lang)
        elif 'ut' in task.lower() or 'test' in task.lower():
            base += "\n" + self.prompts.get('ut_expand', '').format(query=task, lang=lang)
        elif mode == 'doc':
            base += "\n" + self.prompts.get('doc_optimize', '').format(content=task)
        elif '/requirements' in task:
            base += "\n" + self.prompts.get('requirements_gen', '').format(query=task)
        elif '/design' in task:
            base += "\n" + self.prompts.get('design_gen', '').format(query=task, lang=lang)
        elif '/optimize' in task:
            base += "\n" + self.prompts.get('optimize_task', '').format(type=mode, query=task, lang=lang)
        elif '/create-pr' in task:
            base += "\n" + self.prompts.get('create_pr', '').format(query=task)
        elif '/review-pr' in task:
            base += "\n" + self.prompts.get('review_pr', '').format(query=task, lang=lang)
        elif '/commit-push-pr' in task:
            base += "\n" + self.prompts.get('commit_push_pr', '').format(query=task, lang=lang)
        else:
            base += "\n" + self.prompts.get('code_plan', '').format(query=task, lang=lang)

        return base

    def optimize_prompt(self, original_prompt):
        """Use LLM to optimize the prompt for better effectiveness."""
        if 'optimize_prompt' not in self.prompts:
            return original_prompt

        opt_prompt = self.prompts['optimize_prompt'].format(original=original_prompt)
        response = self.call_llm(opt_prompt)

        if response and not response.startswith("Error:"):
            # Extract improved prompt from response
            improved = response.split('Improved prompt:')[-1].strip() if 'Improved prompt:' in response else response
            return improved
        else:
            return original_prompt

    def inject_rag_results(self, paths):
        injected = []
        for path in paths:
            try:
                if os.path.getsize(path) < 5000:
                    # 添加编码处理
                    encodings = ['utf-8', 'gbk', 'latin-1']
                    content = None
                    for encoding in encodings:
                        try:
                            with open(path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    # 最后尝试忽略错误
                    if content is None:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                else:
                    # 对于大文件，先读取再压缩
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
                    content = self.context_manager.compress(content)

                injected.append(f"File: {path}\n{content}")

            except Exception as e:
                print(f"Warning: Failed to inject {path}: {e}")
                continue

        return '\n'.join(injected)

    def call_llm(self, prompt):
        """Enhanced LLM call with NVIDIA thinking model support and retry logic"""
        import time

        # 添加延迟避免频率限制
        time.sleep(0.5)

        if self.context_manager.token_monitor(prompt) > self.config['llm']['max_tokens']:
            print("Debug: Prompt too long, compressing...")
            prompt = self.context_manager.compress(prompt)

        # 对于思维模型，使用更大的 max_tokens
        max_tokens = self.config['llm']['max_tokens']
        if 'thinking' in self.config['llm']['model'].lower():
            max_tokens = min(32768, max_tokens * 2)  # 思维模型需要更多tokens

        payload = {
            'model': self.config['llm']['model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.config['llm']['temperature'],
            'max_tokens': max_tokens,
            'stream': False
        }

        headers = {
            'Content-Type': 'application/json'
        }
        if self.config['llm']['api_key']:
            headers['Authorization'] = f'Bearer {self.config["llm"]["api_key"]}'

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"Debug: LLM request attempt {attempt + 1}/{max_retries}")

                response = requests.post(
                    self.config['llm']['endpoint'],
                    json=payload,
                    headers=headers,
                    timeout=self.config.get('timeouts', {}).get('llm_request', 300)
                )

                print(f"Debug: Response status: {response.status_code}")

                if response.status_code == 401:
                    print("Debug: 401 Unauthorized - checking if rate limiting...")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Debug: Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()

                response_data = response.json()
                choices = response_data.get('choices', [])

                if not choices:
                    print("Debug: No choices in response")
                    return "Error: No response from LLM"

                choice = choices[0]
                message = choice.get('message', {})

                # 修复：处理 NVIDIA 思维模型的响应格式
                content = message.get('content', '')

                # 如果 content 为空或 null，尝试 reasoning_content
                if not content or content == 'null' or content is None:
                    reasoning_content = message.get('reasoning_content', '')
                    if reasoning_content:
                        print(f"Debug: Using reasoning_content (length: {len(reasoning_content)})")
                        content = reasoning_content
                    else:
                        print(f"Debug: Both content and reasoning_content are empty")

                # 检查是否被截断
                finish_reason = choice.get('finish_reason', '')
                if finish_reason == 'length':
                    print("Warning: Response was truncated due to max_tokens limit")
                    print("Consider increasing max_tokens in config.yaml")

                if not content:
                    print(f"Debug: Empty content. Message keys: {list(message.keys())}")
                    print(f"Debug: Full choice: {choice}")
                    return "Error: Empty response from LLM"

                print(f"Debug: Successfully got response (length: {len(content)})")
                return content

            except requests.exceptions.HTTPError as e:
                print(f"Debug: HTTP error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Debug: Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return f"Error: HTTP error after {max_retries} attempts: {e}"

            except requests.exceptions.Timeout:
                print(f"Debug: Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Debug: Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return "Error: Request timeout after multiple attempts"

            except Exception as e:
                print(f"Debug: Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Debug: Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return f"Error: Unexpected error after {max_retries} attempts: {e}"

        return "Error: Failed to get response after all retry attempts"

    def parse_output(self, response):
        """Enhanced output parsing with better error handling"""
        if not response or not response.strip():
            return {'plan': 'Empty LLM response', 'actions': []}

        if response.startswith("Error:"):
            return {'plan': response, 'actions': []}

        try:
            # 尝试直接解析JSON
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # JSON解析失败，尝试从文本中提取信息
        print("Debug: JSON parsing failed, trying text extraction")

        lines = response.split('\n')
        plan = ''
        actions = []
        in_actions = False
        current_action = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 查找计划信息
            if line.lower().startswith('plan:') or '计划:' in line or '设计:' in line:
                plan = line.split(':', 1)[1].strip() if ':' in line else line
                continue

            # 查找动作信息
            if line.lower().startswith('actions:') or '动作:' in line:
                in_actions = True
                continue

            if in_actions and line.strip():
                try:
                    # 尝试解析单行JSON
                    action = json.loads(line)
                    actions.append(action)
                except json.JSONDecodeError:
                    # 如果不是JSON，尝试构建默认动作
                    if 'file_write' in line or 'write' in line.lower():
                        current_action = {
                            'type': 'file_write',
                            'path': 'output.txt',
                            'content': line
                        }
                        actions.append(current_action)

        # 如果没有找到明确的计划，使用响应的前一部分作为计划
        if not plan:
            plan = response[:300] + "..." if len(response) > 300 else response

        # 如果没有找到动作，为设计任务创建默认的文件写入动作
        if not actions and (
                '/design' in getattr(self, 'current_task', '') or 'design' in plan.lower() or '设计' in plan):
            actions = [{
                'type': 'file_write',
                'path': 'design_document.md',
                'content': f"# 设计文档\n\n{response}"
            }]
        elif not actions:
            # 其他情况下，创建一个通用的输出动作
            actions = [{
                'type': 'file_write',
                'path': 'output.md',
                'content': response
            }]

        return {'plan': plan, 'actions': actions}

    def detect_language(self, task):
        """Enhanced language detection"""
        exts = {
            '.py': 'python',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.c': 'c',
            '.js': 'js',
            '.jsx': 'js',
            '.ts': 'js',
            '.tsx': 'js',
            '.java': 'java',
            '.cs': 'cs',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php'
        }

        for ext, l in exts.items():
            if ext in task:
                return l

        # 基于关键词检测
        keywords = {
            'python': ['python', 'py', 'django', 'flask', 'pandas'],
            'js': ['javascript', 'js', 'node', 'react', 'vue'],
            'cpp': ['cpp', 'c++', 'iostream', 'std::'],
            'java': ['java', 'class', 'public static'],
            'go': ['go', 'golang', 'func main'],
            'rust': ['rust', 'cargo', 'fn main'],
        }

        task_lower = task.lower()
        for lang, words in keywords.items():
            if any(word in task_lower for word in words):
                return lang

        return self.config['languages']['default']

    def run_sub_agents(self, sub_tasks, mode, lang):
        """Run multiple sub-agents in parallel"""
        with multiprocessing.Pool(processes=min(len(sub_tasks), 4)) as pool:
            return pool.starmap(self.infer, [(t, mode, lang) for t in sub_tasks])