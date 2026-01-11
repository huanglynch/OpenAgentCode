# executor.py (代码执行器) - 跨平台兼容版本
import os
import shutil
import subprocess
from git import Repo
from tools import ToolLoader
import re
import sys

# 跨平台资源限制：仅在 Unix 系统导入 resource
RESOURCE_AVAILABLE = False
if sys.platform != 'win32':
    try:
        import resource

        RESOURCE_AVAILABLE = True
    except ImportError:
        pass


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
                from git import Repo  # Lazy import
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
                # 修改: 用沙箱执行 bash
                return self.sandbox_exec(self.bash_exec, action['command'])
            elif atype == 'compile_run' or atype == 'run':  # 支持'run'作为别名
                # 修改: 用沙箱执行 compile_run
                return self.sandbox_exec(self.compile_run, action['file'], lang)
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
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
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
        if not self.config['permissions']['exec_bash'] and command.split()[0] not in self.allowed_bash:
            return 'Permission denied: bash command not allowed'
        try:
            # 输入验证（禁止危险关键词）
            dangerous_cmds = ['rm -rf', 'dd if=', 'mkfs', 'wget', 'curl', 'sudo', 'format', 'del /s']
            if any(cmd in command.lower() for cmd in dangerous_cmds):
                return 'Command blocked: potentially dangerous'

            timeout = self.timeouts.get('bash_exec', 300)

            # 跨平台执行：Windows 使用 cmd，Unix 使用 shell
            if sys.platform == 'win32':
                # Windows: 不支持 preexec_fn，直接执行
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='ignore'
                )
            else:
                # Unix: 可以设置资源限制
                def set_limits():
                    if RESOURCE_AVAILABLE:
                        resource.setrlimit(resource.RLIMIT_CPU, (timeout, -1))  # CPU秒
                        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))  # 内存512MB

                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='ignore',
                    preexec_fn=set_limits if RESOURCE_AVAILABLE else None
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
            'cpp': f'g++ {file} -o a.out && ./a.out' if sys.platform != 'win32' else f'g++ {file} -o a.exe && a.exe',
            'c': f'gcc {file} -o a.out && ./a.out' if sys.platform != 'win32' else f'gcc {file} -o a.exe && a.exe',
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
            'cpp': f'./{file}' if sys.platform != 'win32' else file,
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

    def sandbox_exec(self, func, *args, **kwargs):
        """执行高风险函数在临时沙箱目录中"""
        sandbox_dir = 'sandbox_temp'
        original_cwd = os.getcwd()
        try:
            # 使用绝对路径创建沙箱目录
            sandbox_abs = os.path.join(original_cwd, sandbox_dir)
            os.makedirs(sandbox_abs, exist_ok=True)

            # 复制文件到沙箱（在 chdir 之前）
            if len(args) > 0 and isinstance(args[0], str):
                file_path = args[0]
                source = os.path.join(original_cwd, file_path)
                if os.path.exists(source):
                    # 目标路径：沙箱目录内，保持相同的相对路径结构
                    dest_path = os.path.join(sandbox_abs, os.path.basename(file_path))
                    dest_dir = os.path.dirname(dest_path)
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(source, dest_path)

            # 切换到沙箱目录
            os.chdir(sandbox_abs)

            # 修改 args 中的文件路径为相对于沙箱的路径
            new_args = list(args)
            if len(new_args) > 0 and isinstance(new_args[0], str):
                new_args[0] = os.path.basename(new_args[0])

            result = func(*new_args, **kwargs)
            return result
        except Exception as e:
            return f"Sandbox execution failed: {e}"
        finally:
            # 确保恢复原始目录
            os.chdir(original_cwd)
            try:
                shutil.rmtree(os.path.join(original_cwd, sandbox_dir))
            except Exception:
                print("Warning: Failed to clean sandbox")
            # 如果目录仍存在且空，尝试 rmdir
            sandbox_abs = os.path.join(original_cwd, sandbox_dir)
            if os.path.exists(sandbox_abs) and not os.listdir(sandbox_abs):
                try:
                    os.rmdir(sandbox_abs)
                except Exception:
                    pass


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