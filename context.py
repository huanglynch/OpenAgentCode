import yaml
import tiktoken
import os
# 定义默认提示词（避免循环导入）
DEFAULT_PROMPTS = {
    'summarize': """
Compress the following YAML context to key points under {token_limit} tokens while preserving valid YAML format:
{content}
Output only the compressed valid YAML.
"""
}
class ContextManager:
    def __init__(self, config):
        self.config = config
        # 使用原始目录的 context 文件
        original_cwd = config.get('_original_cwd', '.')
        context_file = config['paths']['context_file']
        self.context_file = os.path.join(original_cwd, context_file)
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.max_tokens = self.config['llm']['max_tokens'] // 2
    def load(self):
        if not os.path.exists(self.context_file):
            return {'overview': '', 'history': [], 'facts': [], 'langs': []}
        with open(self.context_file, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            return yaml.safe_load(content) or {'overview': '', 'history': [], 'facts': [], 'langs': []}
        except yaml.YAMLError as e:
            print(f"YAML load error: {e}. Falling back to default context.")
            return {'overview': '', 'history': [], 'facts': [], 'langs': []}
    def update(self, new_data):
        current = self.load()
        if 'task' in new_data:
            current['history'].append(new_data['task'])
        if 'plan' in new_data:
            current['overview'] += f"\n{new_data['plan']}"
        if 'results' in new_data:
            current['facts'].append(new_data['results'])
        if len(current['history']) > 10:
            current['history'] = current['history'][-10:]
        compressed = self.compress(yaml.dump(current))
        with open(self.context_file, 'w', encoding='utf-8') as f:
            f.write(compressed)
    def compress(self, content):
        tokens = self.token_monitor(content)
        if tokens <= self.max_tokens:
            return content
        # 避免循环导入 - 直接调用 LLM
        import requests
        summarize_prompt = DEFAULT_PROMPTS['summarize'].format(
            token_limit=self.max_tokens,
            content=content
        )
        payload = {
            'model': self.config['llm']['model'],
            'messages': [{'role': 'user', 'content': summarize_prompt}],
            'max_tokens': self.max_tokens,
            'stream': False
        }
        headers = {}
        if self.config['llm'].get('api_key', ''):
            headers['Authorization'] = f'Bearer {self.config["llm"]["api_key"]}'
        try:
            response = requests.post(self.config['llm']['endpoint'], json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', content)
        except:
            # 如果 LLM 调用失败，使用简单截断
            return content[:self.max_tokens * 4] # 粗略估计 token 比例
    def clear(self):
        with open(self.context_file, 'w') as f:
            yaml.dump({'overview': '', 'history': [], 'facts': [], 'langs': []}, f)
    def token_monitor(self, content):
        return len(self.encoding.encode(content))