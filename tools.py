# tools.py (工具扩展) - 使用配置的超时值
import os
import importlib.util
import subprocess
from git import Repo
import shutil

try:
    from github import Github
except ImportError:
    Github = None
    print("Warning: PyGithub not installed. GitHub features disabled.")


class ToolLoader:
    def __init__(self, tools_dir, config=None):
        self.config = config or {}

        # 使用原始目录的 tools 目录
        original_cwd = self.config.get('_original_cwd', '.')
        if not os.path.isabs(tools_dir):
            tools_dir = os.path.join(original_cwd, tools_dir)

        self.tools_dir = tools_dir

        # 获取超时配置
        self.timeouts = self.config.get('timeouts', {
            'compile': 300,
            'unit_test': 300,
            'tool_execution': 300
        })

        self.tools = {}
        self.load_tools()

        self.github = None
        if self.config.get('permissions', {}).get('github_api', False) and self.config.get('github', {}).get('token'):
            if Github is None:
                print("GitHub library not available.")
            else:
                self.github = Github(self.config['github']['token'])

    def load_tools(self):
        # 确保目录存在
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir, exist_ok=True)
            print(f"Created tools directory: {self.tools_dir}")
        # 加载自定义工具
        try:
            for filename in os.listdir(self.tools_dir):
                if filename.endswith('.py'):
                    path = os.path.join(self.tools_dir, filename)
                    module_name = filename[:-3]
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if callable(attr) and not attr_name.startswith('__'):
                                self.tools[attr_name] = attr
                        print(f"Loaded tool module: {module_name}")
                    except Exception as e:
                        print(f"Failed to load tool {filename}: {e}")
        except Exception as e:
            print(f"Error loading tools: {e}")
        # Built-in tools
        self.tools['git_clone'] = self.git_clone
        self.tools['file_search'] = self.file_search
        self.tools['compile_lang'] = self.compile_lang
        self.tools['run_ut'] = self.run_ut
        self.tools['create_pr'] = self.create_pr # 新
        self.tools['review_pr'] = self.review_pr # 新

    def get_tool(self, name):
        return self.tools.get(name, lambda **kwargs: f'Tool not found: {name}')

    @staticmethod
    def git_clone(url, dest='.'):
        try:
            Repo.clone_from(url, dest)
            return f'Successfully cloned {url} to {dest}'
        except Exception as e:
            return f'Failed to clone: {e}'

    @staticmethod
    def file_search(pattern):
        matches = []
        try:
            for root, _, files in os.walk('.'):
                for file in files:
                    if pattern in file:
                        matches.append(os.path.join(root, file))
            return matches if matches else f'No files found matching: {pattern}'
        except Exception as e:
            return f'Search failed: {e}'

    # 修改后的 compile_lang 函数（完整，原有代码保留，仅每个分支添加 shutil.which 检查）
    def compile_lang(self, file, lang):
        """Compile source file based on language"""
        timeout = self.timeouts.get('compile', 300)
        try:
            if lang == 'cpp':
                if not shutil.which('g++'):  # 添加：检查编译器
                    return 'Compiler not found: g++'
                result = subprocess.run(['g++', file, '-o', 'a.out'],
                                        capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    return 'Compiled successfully'
                else:
                    return f'Compilation failed: {result.stderr}'
            elif lang == 'c':
                if not shutil.which('gcc'):  # 添加：检查编译器
                    return 'Compiler not found: gcc'
                result = subprocess.run(['gcc', file, '-o', 'a.out'],
                                        capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    return 'Compiled successfully'
                else:
                    return f'Compilation failed: {result.stderr}'
            elif lang == 'java':
                if not shutil.which('javac'):  # 添加：检查编译器
                    return 'Compiler not found: javac'
                result = subprocess.run(['javac', file],
                                        capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    return 'Compiled successfully'
                else:
                    return f'Compilation failed: {result.stderr}'
            elif lang == 'cs':
                if not shutil.which('csc'):  # 添加：检查编译器
                    return 'Compiler not found: csc'
                result = subprocess.run(['csc', file],
                                        capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    return 'Compiled successfully'
                else:
                    return f'Compilation failed: {result.stderr}'
            else:
                return f'Compilation not supported for language: {lang}'
        except subprocess.TimeoutExpired:
            return f'Compilation timeout ({timeout}s)'
        except FileNotFoundError:
            return f'Compiler not found for {lang}'
        except Exception as e:
            return f'Compilation error: {e}'

    def run_ut(self, file, lang):
        """Run unit tests based on language"""
        timeout = self.timeouts.get('unit_test', 300)
        try:
            if lang == 'python':
                result = subprocess.run(['pytest', file, '-v'],
                                        capture_output=True, text=True, timeout=timeout)
                return result.stdout if result.returncode == 0 else result.stderr
            elif lang == 'cpp':
                # Assume executable exists
                if os.path.exists(file):
                    result = subprocess.run([f'./{file}'],
                                            capture_output=True, text=True, timeout=timeout)
                    return result.stdout if result.returncode == 0 else result.stderr
                else:
                    return f'Test executable not found: {file}'
            elif lang == 'java':
                # Assume JUnit setup
                classname = os.path.splitext(os.path.basename(file))[0]
                result = subprocess.run(['java', '-cp', '.:junit.jar',
                                         'org.junit.runner.JUnitCore', classname],
                                        capture_output=True, text=True, timeout=timeout)
                return result.stdout if result.returncode == 0 else result.stderr
            else:
                return f'Unit test not supported for language: {lang}'
        except subprocess.TimeoutExpired:
            return f'Test timeout ({timeout}s)'
        except FileNotFoundError:
            return f'Test runner not found for {lang}'
        except Exception as e:
            return f'Test execution error: {e}'

    def create_pr(self, title, body, head_branch, base_branch='main', owner=None, repo=None):
        if not self.config['permissions'].get('github_api', False):
            return 'Permission denied: github_api'
        if not self.github:
            return 'GitHub not configured'
        owner = owner or self.config['github'].get('owner', '')
        repo = repo or self.config['github'].get('repo', '')
        if not owner or not repo:
            return 'Missing owner/repo in config'
        try:
            gh_repo = self.github.get_repo(f"{owner}/{repo}")
            pr = gh_repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
            return f'PR created: {pr.html_url}'
        except Exception as e:
            return f'Create PR failed: {e}'

    def review_pr(self, pr_number, comment, event='COMMENT', owner=None, repo=None):
        if not self.config['permissions'].get('github_api', False):
            return 'Permission denied: github_api'
        if not self.github:
            return 'GitHub not configured'
        owner = owner or self.config['github'].get('owner', '')
        repo = repo or self.config['github'].get('repo', '')
        if not owner or not repo:
            return 'Missing owner/repo in config'
        try:
            gh_repo = self.github.get_repo(f"{owner}/{repo}")
            pr = gh_repo.get_pull(pr_number)
            pr.create_review(body=comment, event=event)
            return f'PR {pr_number} reviewed: {comment[:100]}...'
        except Exception as e:
            return f'Review PR failed: {e}'