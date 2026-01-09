# agent.py (AI 代理核心) - Full implementation with NVIDIA thinking model support
import requests
import json
from rag_vector import VectorRAG
from context import ContextManager
import multiprocessing
import os
import yaml
import time # 新增: 用于限流
import datetime # 新增: 用于时间戳文件名


class Agent:
    def __init__(self, config, prompts, context_manager, session_file=None):
        self.config = config
        self.prompts = prompts
        self.context_manager = context_manager
        self.rag = VectorRAG(config)
        self.last_call_time = 0
        self.qps = self.config['llm'].get('qps', 0.333)
        self.max_output_display = self.config.get('optimization', {}).get('max_output_display', 120)
        self.max_iterations = self.config.get('optimization', {}).get('max_iterations', 0)
        self.max_react_iterations = self.config.get('optimization', {}).get('max_react_iterations', 3)
        self.session_file = session_file
        # 新增: 获取项目根目录
        self.original_cwd = config.get('_original_cwd', '.')
        # 修改: 使用绝对路径
        self.metrics_log = os.path.join(self.original_cwd, 'logs', 'oac_metrics.json')
        os.makedirs(os.path.join(self.original_cwd, 'logs'), exist_ok=True)
        self.metrics = []

    def infer(self, task, mode='code', lang=None, depth=0): # Added depth param
        if depth > 5: # Prevent recursion depth issues
            return {'plan': 'Max recursion depth reached', 'output': 'Aborted sub-tasks'}
        self.current_depth = depth # Track recursion (new line)
        self.current_task = task # 新增: 记录当前任务用于文件日志
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
        # 新增: 检查任务复杂度，如果简单，直接使用单次推理路径
        if mode in ['auto', 'debug']:
            if 'simple' in task.lower() or len(task) < 50: # 简单任务条件
                # 复制原有单次推理逻辑（避免ReAct循环）
                prompt = self.build_prompt(task, mode, lang, context, injected_rag)
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
                output_str = '\n'.join(results)
                self.save_result_to_file(output_str)
                display_output = output_str[:self.max_output_display] + '...' if len(
                    output_str) > self.max_output_display else output_str
                if self.current_depth == 0: # 只在主任务打印到屏幕
                    print(f"Agent Output (truncated to {self.max_output_display} chars): {display_output}")
                update_data = {
                    'task': task,
                    'plan': parsed.get('plan', ''),
                    'results': output_str
                }
                self.context_manager.update(update_data)
                # 迭代优化逻辑（受max_iterations控制）
                iteration_count = 0
                while self.max_iterations > 0 and iteration_count < self.max_iterations:
                    if self.check_result(output_str):
                        break
                    print(f"Iteration {iteration_count + 1}: Result check failed, optimizing...")
                    optimized_task = f"Debug and fix error in previous output: {output_str[:500]}"
                    optimized_result = self.infer(optimized_task, mode='debug', lang=lang, depth=depth + 1)
                    output_str = optimized_result.get('output', output_str)
                    self.save_result_to_file(output_str)
                    display_output = output_str[:self.max_output_display] + '...' if len(
                        output_str) > self.max_output_display else output_str
                    if self.current_depth == 0: # 只在主任务打印到屏幕
                        print(f"Optimized Output (truncated): {display_output}")
                    update_data['results'] = output_str
                    self.context_manager.update(update_data)
                    iteration_count += 1
                result = {
                    'plan': parsed.get('plan', ''),
                    'output': output_str
                }
            else:
                result = self.react_loop(task, mode, lang, context, injected_rag, depth)
        else:
            # 原有单次推理逻辑（保持不变）
            prompt = self.build_prompt(task, mode, lang, context, injected_rag)
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
            output_str = '\n'.join(results)
            self.save_result_to_file(output_str)
            display_output = output_str[:self.max_output_display] + '...' if len(
                output_str) > self.max_output_display else output_str
            if self.current_depth == 0: # 只在主任务打印到屏幕
                print(f"Agent Output (truncated to {self.max_output_display} chars): {display_output}")
            update_data = {
                'task': task,
                'plan': parsed.get('plan', ''),
                'results': output_str
            }
            self.context_manager.update(update_data)
            iteration_count = 0
            while self.max_iterations > 0 and iteration_count < self.max_iterations:
                if self.check_result(output_str):
                    break
                print(f"Iteration {iteration_count + 1}: Result check failed, optimizing...")
                optimized_task = f"Debug and fix error in previous output: {output_str[:500]}"
                optimized_result = self.infer(optimized_task, mode='debug', lang=lang, depth=depth + 1)
                output_str = optimized_result.get('output', output_str)
                self.save_result_to_file(output_str)
                display_output = output_str[:self.max_output_display] + '...' if len(
                    output_str) > self.max_output_display else output_str
                if self.current_depth == 0: # 只在主任务打印到屏幕
                    print(f"Optimized Output (truncated): {display_output}")
                update_data['results'] = output_str
                self.context_manager.update(update_data)
                iteration_count += 1
            result = {
                'plan': parsed.get('plan', ''),
                'output': output_str
            }
        # 新增: 记录指标并保存日志
        success = self.check_result(result['output'])
        metrics_entry = {
            'task': task,
            'mode': mode,
            'iterations': iteration_count if 'iteration_count' in locals() else 0,
            'success': success,
            'depth': depth
        }
        self.metrics.append(metrics_entry)
        self.save_metrics()
        return result

    # 新增: ReAct 循环方法
    def react_loop(self, task, mode, lang, context, injected_rag, depth):
        """ReAct 框架：Reason + Act + Observe 循环，直到任务完成"""
        observations = [] # 存储观察结果
        react_iteration = 0
        output_str = ''
        plan = ''
        while react_iteration < self.max_react_iterations:
            # 构建 ReAct 提示：注入先前观察
            react_prompt = self.build_prompt(task, mode, lang, context, injected_rag)
            react_prompt += "\nReAct: Think step-by-step, then output actions. If task complete, set 'done': true in JSON."
            if observations:
                react_prompt += f"\nPrevious observations: {yaml.dump(observations[-3:])}" # 仅最后3个观察，避免 token 爆炸
            response = self.call_llm(react_prompt)
            if not response or response.startswith("Error:"):
                return {'plan': 'LLM call failed', 'output': response}
            parsed = self.parse_output(response)
            # 执行 actions
            from executor import get_executor
            executor = get_executor(mode, self.config)
            results = []
            for action in parsed.get('actions', []):
                try:
                    result = executor.execute(action, lang=lang)
                    results.append(result)
                except Exception as e:
                    results.append(f"Action failed: {e}")
            observation = '\n'.join(results)
            observations.append(observation)
            output_str += observation + '\n'
            plan = parsed.get('plan', plan) # 更新 plan
            # 检查是否完成（LLM 输出 'done': true）
            if parsed.get('done', False):
                break
            react_iteration += 1
        # 保存和更新（如原有逻辑）
        self.save_result_to_file(output_str)
        display_output = output_str[:self.max_output_display] + '...' if len(output_str) > self.max_output_display else output_str
        if self.current_depth == 0: # 只在主任务打印到屏幕
            print(f"ReAct Output (truncated): {display_output}")
        update_data = {'task': task, 'plan': plan, 'results': output_str}
        self.context_manager.update(update_data)
        return {'plan': plan, 'output': output_str}

    # 新增: 检查结果是否明显错误的方法（简单关键词检查，可扩展）
    def check_result(self, output_str):
        error_keywords = ["error:", "failed:", "exception:", "aborted", "permission denied"]
        return not any(keyword.lower() in output_str.lower() for keyword in error_keywords)

    def save_result_to_file(self, output_str):
        # 修改: 使用绝对路径
        result_dir = os.path.join(self.original_cwd, 'result')
        os.makedirs(result_dir, exist_ok=True)
        if self.session_file:
            header = f"--- Sub-task: {self.current_task} (depth {self.current_depth}) ---" if self.current_depth > 0 else f"--- New Result: {self.current_task} ---"
            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{header}\n{output_str}\n")
            if self.current_depth == 0:
                print(f"Result appended to session file: {self.session_file}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"OAC_RESULT_{timestamp}.txt"
            filepath = os.path.join(result_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(output_str)
            print(f"Result saved to: {filepath}")

    # 新增: 保存指标日志
    def save_metrics(self):
        """追加指标到 JSON 文件"""
        try:
            existing = []
            if os.path.exists(self.metrics_log):
                with open(self.metrics_log, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            existing.extend(self.metrics)
            with open(self.metrics_log, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2)
            self.metrics = [] # 清空临时存储
            print(f"Metrics logged to {self.metrics_log}")
        except Exception as e:
            print(f"Metrics logging failed: {e}") # 新增: 评估指标

    def build_prompt(self, task, mode, lang, context, injected_rag):
        base = self.prompts.get('base_prompt', '').format(
            mode=mode,
            lang=lang,
            context=yaml.dump(context) + "\n" + injected_rag
        )
        # 新增：强制JSON输出格式，并提供动作示例
        base += "\nAlways output in JSON format: {'plan': 'your plan', 'actions': [{'type': 'file_write', 'path': 'file.py', 'content': 'code'}, {'type': 'compile_run', 'file': 'file.py', 'lang': 'python'}]}"
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
        # New: Append decomposition guidance if mode=='auto'
        if mode == 'auto':
            base += "\n" + self.prompts.get('decompose_task', '').format(query=task)
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
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                else:
                    # 对于大文件，先读取再压缩
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
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
        # 新增: QPS 限流逻辑
        current_time = time.time()
        if self.qps > 0:
            min_interval = 1 / self.qps
            delay = max(0, min_interval - (current_time - self.last_call_time))
            if delay > 0:
                print(f"Debug: Rate limiting - sleeping for {delay:.2f} seconds")
                time.sleep(delay)
        self.last_call_time = time.time()
        # 添加延迟避免频率限制（原有）
        time.sleep(0.5)
        if self.context_manager.token_monitor(prompt) > self.config['llm']['max_tokens']:
            print("Debug: Prompt too long, compressing...")
            prompt = self.context_manager.compress(prompt)
        # 对于思维模型，使用更大的 max_tokens
        max_tokens = self.config['llm']['max_tokens']
        if 'thinking' in self.config['llm']['model'].lower():
            max_tokens = min(32768, max_tokens * 2) # 思维模型需要更多tokens
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
                # New: Handle sub_tasks if present
                sub_tasks = parsed.get('sub_tasks', [])
                if sub_tasks:
                    sub_results = [self.infer(t, mode='auto', lang=self.detect_language(t), depth=self.current_depth + 1) for t in sub_tasks] # 顺序执行子任务
                    # 合并结果（按顺序）
                    sub_outputs = [r.get('output', '') for r in sub_results]
                    # 直接附加到plan作为描述（不作为action）
                    parsed[
                        'plan'] += f"\nSub-tasks executed: {len(sub_tasks)}\nSub-results:\n{' '.join(sub_outputs)}" # <-- 改这里
                # 新增: 检查 ReAct 'done' 字段
                parsed['done'] = parsed.get('done', False)
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
            # 新增：智能推断执行动作，如果响应包含代码且任务涉及运行
            import re
            if 'print(' in response and ('run' in self.current_task.lower() or '执行' in self.current_task.lower()):
                code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    temp_path = 'temp_exec.py'
                    actions.insert(0, {'type': 'file_write', 'path': temp_path, 'content': code})
                    actions.append({'type': 'compile_run', 'file': temp_path, 'lang': 'python'})
        # 新增：如果响应提到'execute'或'run'，添加compile_run（假设文件已写）
        if ('execute' in response.lower() or 'run' in response.lower()) and any(
                a['type'] == 'file_write' for a in actions):
            file_path = actions[0].get('path', 'output.py') # 假设第一个write是代码文件
            actions.append({'type': 'compile_run', 'file': file_path, 'lang': 'python'})
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