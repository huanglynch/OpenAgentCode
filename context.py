import yaml
import tiktoken
import os
# 定义默认提示词（避免循环导入）
DEFAULT_PROMPTS = {
    'summarize': """
Compress to key points (< {token_limit} tokens): {content}
"""
}
class ContextManager:
    def __init__(self, config):
        self.config = config
        self.context_file = self.config['paths']['context_file']
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.max_tokens = self.config['llm']['max_tokens'] // 2
    def load(self):
        if not os.path.exists(self.context_file):
            return {'overview': '', 'history': [], 'facts': [], 'langs': []}
        with open(self.context_file, 'r') as f:
            content = f.read()
        return yaml.safe_load(content) or {'overview': '', 'history': [], 'facts': [], 'langs': []}
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
        with open(self.context_file, 'w') as f:
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