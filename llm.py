# llm.py - LLM client wrapper
import requests
import time


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.endpoint = config.get('endpoint', 'https://api.openai.com/v1/chat/completions')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)

    def chat(self, messages, temperature=None, max_tokens=None):
        """Send chat request to LLM"""
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature or self.temperature,
            'max_tokens': max_tokens or self.max_tokens,
            'stream': False
        }

        headers = {
            'Content-Type': 'application/json'
        }

        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=300
                )

                if response.status_code == 401 and attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue

                response.raise_for_status()

                data = response.json()
                choices = data.get('choices', [])

                if not choices:
                    return "Error: No response from LLM"

                message = choices[0].get('message', {})
                content = message.get('content', '')

                # Handle NVIDIA thinking models
                if not content or content == 'null' or content is None:
                    content = message.get('reasoning_content', '')

                if not content:
                    return "Error: Empty response from LLM"

                return content

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    return f"Error: {e}"

        return "Error: Failed after all retries"