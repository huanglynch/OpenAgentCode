import os
import shutil
import subprocess
from git import Repo
from tools import ToolLoader
import re
def load_config():
    import yaml
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        # 返回默认配置
        return {
            'timeouts': {
                'bash_exec': 300,
                'compile': 300,
                'unit_test': 300,
                'llm_request': 300,
                'tool_execution': 300
            }
        }

class BaseExecutor:
    def execute(self, action, lang=None):
        raise NotImplementedError

class CodeExecutor(BaseExecutor):
    def __init__(self, config):
        self.config = config
        self.timeouts = self.config.get('timeouts', {
            'bash_exec': 300,
            'compile': 300,
            'unit_test': 300,
            'llm_request': 300,
            'tool_execution': 300
        })
        # 新增
        self.allowed_bash = self.config['permissions'].get('allowed_bash_commands', [])
        # 传递 config 给 ToolLoader
        self.tool_loader = ToolLoader(self.config['paths']['tools_dir'], self.config)
        if self.config.get('permissions', {}).get('git_commit', False):
            try:
                from git import Repo # Lazy import
                self.repo = Repo('.')
            except ImportError:
                print("Warning: GitPython library not installed.")
                self.repo = None
            except Exception as e:
                print(f"Warning: Not a git repository or git error: {e}")
                self.repo = None
        else:
            self.repo = None

    def execute(self, action, lang=None):
        atype = action.get('type', '')
        try:
            if atype == 'file_read':
                return self.file_read(action['path'])
            elif atype == 'file_write':
                return self.file_write(action['path'], action['content'])
            elif atype == 'git_commit':
                return self.git_commit(action['message'])
            elif atype == 'bash':
                return self.bash_exec(action['command'])
            elif atype == 'compile_run' or atype == 'run':  # 新增：支持'run'作为别名
                return self.compile_run(action['file'], lang)
            elif atype == 'run_ut':
                return self.run_ut(action['file'], lang)
            elif atype == 'tool':
                tool = self.tool_loader.get_tool(action['name'])
                return tool(**action.get('args', {}))
            elif atype == 'create_pr':
                return self.tool_loader.create_pr(
                    action['title'], action['body'], action.get('head', 'feature'),
                    action.get('base', 'main'), action.get('owner'), action.get('repo')
                )
            elif atype == 'review_pr':
                return self.tool_loader.review_pr(
                    action['pr_number'], action['comment'], action.get('event', 'COMMENT'),
                    action.get('owner'), action.get('repo')
                )
            return f'Unknown action type: {atype}'
        except KeyError as e:
            return f'Missing required field in action: {e}'
        except Exception as e:
            return f'Action execution failed: {e}'

    def file_read(self, path):
        if not self.config['permissions']['file_read']:
            return 'Permission denied: file_read'
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，使用 errors='ignore'
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except FileNotFoundError:
            return f'File not found: {path}'
        except Exception as e:
            return f'Failed to read file: {e}'
    def file_write(self, path, content):
        if not self.config['permissions']['file_write']:
            return 'Permission denied: file_write'
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f'Successfully written to {path}'
        except Exception as e:
            return f'Failed to write file: {e}'
    def git_commit(self, message):
        if not self.repo:
            return 'Not a git repository'
        if not self.config['permissions']['git_commit']:
            return 'Permission denied: git_commit'
        try:
            self.repo.git.add(A=True)
            self.repo.index.commit(message)
            return f'Committed: {message}'
        except Exception as e:
            return f'Git commit failed: {e}'

    def bash_exec(self, command):
        if not self.config['permissions']['exec_bash'] and command not in self.allowed_bash:
            return 'Permission denied: bash command not allowed'
        try:
            timeout = self.timeouts.get('bash_exec', 300)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return output or f'Command executed (return code: {result.returncode})'
        except subprocess.TimeoutExpired:
            return f'Command timeout ({self.timeouts.get("bash_exec", 300)}s)'
        except Exception as e:
            return f'Command execution failed: {e}'

    def compile_run(self, file, lang):
        commands = {
            'python': ['python', file],
            'cpp': f'g++ {file} -o a.out && ./a.out',
            'c': f'gcc {file} -o a.out && ./a.out',
            'js': ['node', file],
            'java': f'javac {file} && java {os.path.splitext(os.path.basename(file))[0]}',
            'cs': f'csc {file} && {os.path.splitext(file)[0]}.exe',
            'go': ['go', 'run', file]
        }
        if lang not in commands:
            return f'Unsupported language: {lang}'
        cmd = commands[lang]
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        # 使用 compile 超时
        timeout = self.timeouts.get('compile', 300)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return output or f'Compiled and executed (return code: {result.returncode})'
        except subprocess.TimeoutExpired:
            return f'Compilation/execution timeout ({timeout}s)'
        except Exception as e:
            return f'Compilation/execution failed: {e}'

    def run_ut(self, file, lang):
        ut_commands = {
            'python': f'pytest {file} -v',
            'cpp': f'./{file}',
            'java': f'java -cp .:junit.jar org.junit.runner.JUnitCore {os.path.splitext(os.path.basename(file))[0]}',
            'js': f'npm test {file}',
            'go': f'go test {file}',
        }
        if lang not in ut_commands:
            return f'Unsupported UT language: {lang}'
        # 使用 unit_test 超时
        timeout = self.timeouts.get('unit_test', 300)
        try:
            result = subprocess.run(
                ut_commands[lang],
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return f'Unit test timeout ({timeout}s)'
        except Exception as e:
            return f'Unit test failed: {e}'

class DocExecutor(BaseExecutor):
    def __init__(self, config):
        self.config = config
    def execute(self, action, lang=None):
        atype = action.get('type', '')
        try:
            if atype == 'summarize':
                return self.summarize(action['content'])
            elif atype == 'optimize':
                return self.optimize(action['content'])
            elif atype == 'extract':
                return self.extract(action['content'], action['pattern'])
            return f'Unknown doc action: {atype}'
        except KeyError as e:
            return f'Missing required field: {e}'
        except Exception as e:
            return f'Doc action failed: {e}'
    def summarize(self, content):
        try:
            lines = content.split('\n')
            summary = [line for line in lines if line.startswith('#') or len(line) > 50]
            return '\n'.join(summary) if summary else 'No summary available'
        except Exception as e:
            return f'Summarize failed: {e}'
    def optimize(self, content):
        try:
            # 去除多余空白
            content = re.sub(r'\s+', ' ', content)
            # 规范化标题
            content = re.sub(r'#+\s*', lambda m: m.group(0).strip() + ' ', content)
            # 去除行尾空白
            content = '\n'.join(line.rstrip() for line in content.split('\n'))
            return content
        except Exception as e:
            return f'Optimize failed: {e}'
    def extract(self, content, pattern):
        try:
            matches = re.findall(pattern, content, re.MULTILINE)
            return '\n'.join(matches) if matches else f'No matches found for pattern: {pattern}'
        except re.error as e:
            return f'Invalid regex pattern: {e}'
        except Exception as e:
            return f'Extract failed: {e}'

def get_executor(mode, config=None):
    if not config:
        config = load_config()
    if mode == 'code' or mode == 'auto' or 'code' in mode:
        return CodeExecutor(config)
    elif mode == 'doc':
        return DocExecutor(config)
    # For other tasks, default to CodeExecutor
    return CodeExecutor(config)