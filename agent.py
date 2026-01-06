# agent.py (AI 代理核心) - Full implementation
import requests
import json
from rag_vector import VectorRAG
from context import ContextManager
import multiprocessing
import os
import yaml # 添加此行
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
        if self.context_manager.token_monitor(yaml.dump(context)) > self.config['llm']['max_tokens'] * 0.5:
            context = self.context_manager.compress(yaml.dump(context))
        rag_results = self.rag.search(task, self.config['rag']['top_k'], mode, lang)
        injected_rag = self.inject_rag_results(rag_results)
        prompt = self.build_prompt(task, mode, lang, context, injected_rag)
        # 自动优化 prompt，如果是 debug 相关
        if 'debug' in task.lower() or 'error' in task.lower():
            prompt = self.optimize_prompt(prompt)
        response = self.call_llm(prompt)
        parsed = self.parse_output(response)
        from executor import get_executor
        executor = get_executor(mode, self.config)
        results = []
        for action in parsed.get('actions', []):
            result = executor.execute(action, lang=lang)
            results.append(result)
        update_data = {'task': task, 'plan': parsed.get('plan', ''), 'results': '\n'.join(results)}
        self.context_manager.update(update_data)
        return {'plan': parsed.get('plan', ''), 'output': '\n'.join(results)}

    def build_prompt(self, task, mode, lang, context, injected_rag):
        base = self.prompts['base_prompt'].format(mode=mode, lang=lang, context=yaml.dump(context) + injected_rag)
        if 'debug' in task.lower():
            base += self.prompts['error_handle'].format(error=task, lang=lang)
        elif 'ut' in task.lower():
            base += self.prompts['ut_expand'].format(query=task, lang=lang)
        elif mode == 'doc':
            base += self.prompts['doc_optimize'].format(content=task)
        elif '/requirements' in task:
            base += self.prompts['requirements_gen'].format(query=task)
        elif '/design' in task:
            base += self.prompts['design_gen'].format(query=task, lang=lang)
        elif '/optimize' in task:
            base += self.prompts['optimize_task'].format(type=mode, query=task, lang=lang)
        elif '/create-pr' in task:
            base += self.prompts['create_pr'].format(query=task)
        elif '/review-pr' in task:
            base += self.prompts['review_pr'].format(query=task, lang=lang)
        elif '/commit-push-pr' in task:  # 新增
            base += self.prompts['commit_push_pr'].format(query=task, lang=lang)
        else:
            base += self.prompts['code_plan'].format(query=task, lang=lang)
        return base

    def optimize_prompt(self, original_prompt):
        """Use LLM to optimize the prompt for better effectiveness."""
        opt_prompt = self.prompts['optimize_prompt'].format(original=original_prompt)
        response = self.call_llm(opt_prompt)
        # Extract improved prompt from response (assume last part is the improved one)
        improved = response.split('Improved prompt:')[-1].strip() if 'Improved prompt:' in response else original_prompt
        return improved

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
        if self.context_manager.token_monitor(prompt) > self.config['llm']['max_tokens']:
            prompt = self.context_manager.compress(prompt)
        payload = {
            'model': self.config['llm']['model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.config['llm']['temperature'],
            'max_tokens': self.config['llm']['max_tokens'],
            'stream': False
        }
        headers = {}
        if self.config['llm']['api_key']:
            headers['Authorization'] = f'Bearer {self.config["llm"]["api_key"]}'
        response = requests.post(self.config['llm']['endpoint'], json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

    def parse_output(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            lines = response.split('\n')
            plan = ''
            actions = []
            in_actions = False
            for line in lines:
                if line.startswith('plan:'):
                    plan = line.split(':', 1)[1].strip()
                elif line.startswith('actions:'):
                    in_actions = True
                elif in_actions and line.strip():
                    try:
                        actions.append(json.loads(line))
                    except:
                        pass
            return {'plan': plan, 'actions': actions}

    def detect_language(self, task):
        exts = {'.py': 'python', '.cpp': 'cpp', '.h': 'cpp', '.c': 'c', '.js': 'js', '.java': 'java', '.cs': 'cs', '.go': 'go'}
        for ext, l in exts.items():
            if ext in task:
                return l
        return self.config['languages']['default']

    def run_sub_agents(self, sub_tasks, mode, lang):
        with multiprocessing.Pool(processes=len(sub_tasks)) as pool:
            return pool.starmap(self.infer, [(t, mode, lang) for t in sub_tasks])