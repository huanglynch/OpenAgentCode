# rag_vector.py - 完整的向量RAG实现
import os
#os.environ['HF_HUB_OFFLINE'] = '1'  # 启用离线模式,转由配置文件控制
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import time
import requests
# Tree-sitter 可选导入
TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("Warning: tree-sitter not available. Using token-based chunking only.")
# Watchdog 可选导入
WATCHDOG_AVAILABLE = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("Warning: watchdog not available. Auto-refresh disabled.")

# 新增: FAISS 可选导入（用于向量搜索加速）
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not available. Using full scan for vector search.")


class VectorRAG:
    def __init__(self, config):
        self.config = config
        # 初始化基础属性 (不变)
        self.vectorizer = CountVectorizer()
        self.alpha = self.config['rag'].get('hybrid_alpha', 0.4)
        if self.config['rag'].get('offline_mode', False):
            os.environ['HF_HUB_OFFLINE'] = '1'
            print("✓ HF_HUB_OFFLINE enabled via config")
        self.index = {}
        self.contents = []
        self.paths = []
        self.faiss_index = None if not FAISS_AVAILABLE else None
        # 获取并规范化缓存目录路径 (不变)
        cache_dir = self._get_cache_dir()
        # 加载模型（修改：使用try-except处理下载）
        self.model = self._load_model(cache_dir)
        # 构建索引 (不变)
        self.build_index()
        # 设置文件监控 (不变)
        refresh_interval = self.config['rag'].get('index_refresh_interval', 0)
        if WATCHDOG_AVAILABLE and refresh_interval > 0:
            self.setup_watcher()

    def _get_cache_dir(self):
        """获取并规范化缓存目录的绝对路径"""
        cache_dir = os.path.expanduser(self.config['paths']['embed_cache_dir'])
        if not os.path.isabs(cache_dir):
            original_cwd = self.config.get('_original_cwd', '.')
            cache_dir = os.path.join(original_cwd, cache_dir)
        # 转换为绝对路径并规范化
        cache_dir = os.path.abspath(cache_dir)
        # 确保目录存在
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _load_model(self, cache_dir):
        """加载嵌入模型: 优先本地缓存; 如果失败, 尝试下载; 下载失败则报错优雅退出。"""
        model_name = self.config['rag']['embedding_model']
        print(f"Attempting to load model '{model_name}' (cache: {cache_dir})")
        try:
            # 先尝试仅用本地缓存加载（优先本地）
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                local_files_only=True  # 强制仅本地
            )
            print(f"✓ Loaded from local cache")
            return model
        except Exception as local_e:
            # 本地失败，fallback到尝试下载
            print(f"Local load failed: {local_e}. Falling back to download.")
            try:
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir  # 会自动下载缺失文件
                )
                print(f"✓ Successfully loaded model (with download)")
                return model
            except Exception as download_e:
                # 两者失败，报错优雅退出
                error_msg = f"Failed to load '{model_name}':\n- Local load error: {local_e}\n- Download error: {download_e}\n\n"
                error_msg += "解决方案:\n"
                error_msg += f"1. 手动下载模型 'sentence-transformers/{model_name}' 到缓存目录: {cache_dir}\n"
                error_msg += "   (从 https://huggingface.co/sentence-transformers/all-mpnet-base-v2 下载文件，并放置到正确子目录)\n"
                error_msg += "2. 或在 config.yaml 中禁用 RAG: 设置 rag.enabled: false\n"
                error_msg += "3. 检查网络/代理设置，或使用离线环境变量: export HF_HUB_OFFLINE=1\n"
                print(error_msg)
                raise RuntimeError(error_msg)  # 抛出，让上层捕获并退出初始化

    def rebuild_index(self):
        """Clear and rebuild the entire index"""
        print("Clearing existing RAG index...")
        self.index.clear()
        self.contents.clear()
        self.paths.clear()
        # 新增: 清空 FAISS 索引
        self.faiss_index = None  # 新增: FAISS 支持
        print("Rebuilding RAG index...")
        self.build_index()
    def build_index(self):
        """Build vector and BM25 index for all files"""
        print("Building RAG index...")
        self.paths = self.scan_files()
        if not self.paths:
            print("Warning: No files found to index")
            return
        self.contents = []
        chunks_per_file = []
        # 新增: 收集所有嵌入用于 FAISS
        all_embs = []  # 新增: FAISS 支持
        for path in self.paths:
            try:
                chunks = self.chunk_file(path, self.config['rag']['chunk_size'])
                if not chunks:
                    continue
                embs = self.model.encode(chunks)
                self.contents.extend(chunks)
                chunks_per_file.append(len(chunks))
                self.index[path] = {'embs': embs, 'chunks': chunks}
                # 新增: 追加嵌入
                all_embs.extend(embs)  # 新增: FAISS 支持
            except Exception as e:
                print(f"Failed to index {path}: {e}")
                continue
        # Build BM25 index
        if self.contents:
            try:
                bm25_matrix = self.vectorizer.fit_transform(self.contents)
                # Assign BM25 vectors per file
                chunk_idx = 0
                for i, path in enumerate(self.paths):
                    if path not in self.index:
                        continue
                    num_chunks = chunks_per_file[i]
                    self.index[path]['bm25'] = bm25_matrix[chunk_idx:chunk_idx + num_chunks]
                    chunk_idx += num_chunks
                print(f"✓ Indexed {len(self.paths)} files with {len(self.contents)} chunks")
            except Exception as e:
                print(f"BM25 indexing failed: {e}")
        # 新增: 构建 FAISS 索引（如果可用）
        if FAISS_AVAILABLE and all_embs:
            try:
                dim = len(all_embs[0])
                self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product for cosine (normalize if needed)
                all_embs_np = np.array(all_embs).astype('float32')
                faiss.normalize_L2(all_embs_np)  # Normalize for cosine similarity
                self.faiss_index.add(all_embs_np)
                print("✓ FAISS index built for fast vector search")
            except Exception as e:
                print(f"FAISS index failed: {e}. Falling back to full scan.")
                self.faiss_index = None  # 新增: FAISS 支持

    # 修改后的 scan_files 函数（完整，原有代码保留，仅添加权限检查）
    def scan_files(self):
        """Scan repository for indexable files"""
        exts = ['.md', '.markdown', '.txt', '.text', '.py', '.cpp', '.h', '.cc',
                '.c', '.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go', '.rs']
        files = []
        try:
            for root, dirs, filenames in os.walk('.'):
                if not self.config['permissions'].get('file_read', False):  # 添加：权限检查，跳过未授权
                    continue
                # 规范化路径
                root_norm = os.path.normpath(root)
                # 跨平台检查隐藏目录
                skip_dirs = ['.git', '.agent', 'build', '__pycache__', 'cached_models',
                             'node_modules', 'venv', '.venv', 'dist', '.rag_index']
                # 移除需要跳过的目录
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                # 检查当前路径是否包含跳过的目录
                if any(skip_dir in root_norm.split(os.path.sep) for skip_dir in skip_dirs):
                    continue
                for file in filenames:
                    if any(file.endswith(ext) for ext in exts):
                        filepath = os.path.join(root, file)
                        files.append(filepath)
        except Exception as e:
            print(f"File scan error: {e}")
        return files

    def search(self, query, top_k, mode, lang):
        """Hybrid search with semantic + BM25"""
        if not self.index:
            print("Warning: Index is empty")
            return []
        # Filter by language if specified
        lang_to_ext = {
            'python': ['.py'],
            'cpp': ['.cpp', '.cc', '.h', '.hpp'],
            'c': ['.c', '.h'],
            'js': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'cs': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
        }
        if lang:
            exts = lang_to_ext.get(lang, [f'.{lang}'])
            paths = [p for p in self.paths
                     if any(p.endswith(ext) for ext in exts) or
                     lang.lower() in os.path.basename(p).lower()]
        else:
            paths = self.paths
        if not paths:
            print(f"No files found for language: {lang}")
            return []
        try:
            # Encode query
            query_emb = self.model.encode([query])[0]
            query_bm25 = self.vectorizer.transform([query])
            scores = {}
            # 修改: 使用 FAISS 加速语义搜索（如果可用）
            if self.faiss_index:
                query_emb_np = np.array([query_emb]).astype('float32')
                faiss.normalize_L2(query_emb_np)
                D, I = self.faiss_index.search(query_emb_np, len(self.contents))  # 全搜索，但 FAISS 高效
                emb_scores_per_chunk = D[0]  # 相似度分数
                chunk_idx = 0
                for path in paths:
                    data = self.index.get(path, {})
                    if not data:
                        continue
                    num_chunks = len(data.get('chunks', []))
                    emb_scores = emb_scores_per_chunk[chunk_idx:chunk_idx + num_chunks]
                    avg_emb_score = np.mean(emb_scores) if len(emb_scores) > 0 else 0
                    chunk_idx += num_chunks
            else:
                # 原有全扫描 fallback
                for path in paths:
                    data = self.index.get(path, {})
                    if not data:
                        continue
                    embs = data.get('embs', [])
                    if len(embs) > 0:
                        emb_scores = [cosine_similarity([query_emb], [emb])[0][0] for emb in embs]
                        avg_emb_score = np.mean(emb_scores)
                    else:
                        avg_emb_score = 0
            # BM25 similarity（不变）
            bm25 = data.get('bm25')
            if bm25 is not None and bm25.shape[0] > 0:
                bm25_scores = cosine_similarity(query_bm25, bm25)[0]
                avg_bm25_score = np.mean(bm25_scores)
            else:
                avg_bm25_score = 0
            # Hybrid score
            hybrid = self.alpha * avg_bm25_score + (1 - self.alpha) * avg_emb_score
            # Boost if query term in filename
            query_lower = query.lower()
            filename_lower = os.path.basename(path).lower()
            if any(term in filename_lower for term in query_lower.split()):
                hybrid += 0.2
            scores[path] = hybrid
            # Rank and return
            ranked = sorted(scores, key=scores.get, reverse=True)[:top_k * 2]
            # Rerank if enabled
            rerank_enabled = self.config['rag'].get('rerank_enabled', False)
            if rerank_enabled and len(ranked) > top_k:
                ranked = self.rerank(ranked, query, mode)
            return ranked[:top_k]
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    def rerank(self, paths, query, mode):
        """LLM-based reranking"""
        try:
            snippets = []
            for path in paths:
                chunks = self.index[path].get('chunks', [])
                snippet = ' '.join(chunks[:3])[:500]
                snippets.append(snippet)
            # Get rerank prompt from config
            task_opts = self.config.get('tasks_optimizations', {})
            mode_opts = task_opts.get(mode, {})
            rerank_prompt = mode_opts.get('rerank_prompt', 'Rank by relevance: {query}')
            rerank_prompt = rerank_prompt.format(query=query, k=len(paths))
            rerank_prompt += '\nSnippets:\n' + '\n'.join(
                [f"{i}: {s}" for i, s in enumerate(snippets)]
            )
            response = self.call_llm(rerank_prompt + '\nOutput ranked indices (space-separated):')
            # Parse indices
            indices = []
            for token in response.split():
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < len(paths):
                        indices.append(idx)
            reranked = [paths[i] for i in indices]
            # Keep original order for unparsed items
            remaining = [p for p in paths if p not in reranked]
            return reranked + remaining
        except Exception as e:
            print(f"Rerank failed: {e}")
            return paths
    def call_llm(self, prompt):
        """Call LLM for reranking"""
        try:
            timeout = self.config.get('timeouts', {}).get('llm_request', 300)
            payload = {
                'model': self.config['llm']['model'],
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 512,
                'stream': False
            }
            headers = {
                'Content-Type': 'application/json'
            }
            api_key = self.config['llm'].get('api_key', '')
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            endpoint = self.config['llm']['endpoint']
            resp = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '')
                # Handle NVIDIA thinking models
                if not content or content == 'null' or content is None:
                    content = message.get('reasoning_content', '')
                return content
            return ''
        except requests.Timeout:
            print(f"LLM request timeout ({timeout}s)")
            return ''
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ''

    def chunk_file(self, path, chunk_size):
        """Chunk file using tree-sitter or token-based fallback"""
        lang_map = {
            'py': 'python',
            'cpp': 'cpp',
            'c': 'c',
            'js': 'javascript',
            'java': 'java',
            'cs': 'c_sharp',
            'go': 'go',
            'rs': 'rust'
        }
        ext = path.split('.')[-1] if '.' in path else ''
        # Try tree-sitter if available
        if TREE_SITTER_AVAILABLE and ext in lang_map:
            try:
                lang_lib = 'build/my-languages.so'
                # Check if library exists
                if os.path.exists(lang_lib):
                    LANGUAGE = Language(lang_lib, lang_map[ext])
                    parser = Parser()
                    parser.set_language(LANGUAGE)
                    with open(path, 'rb') as f:
                        tree = parser.parse(f.read())
                    chunks = []

                    def traverse(node):
                        if node.type in ['function_definition', 'class_definition',
                                         'method_definition', 'function_declaration']:
                            try:
                                chunks.append(node.text.decode('utf-8', errors='ignore'))
                            except:
                                pass
                        for child in node.children:
                            traverse(child)

                    traverse(tree.root_node)
                    if chunks:
                        return chunks
            except Exception as e:
                print(f"Tree-sitter parsing failed for {path}: {e}")
        # Fallback to token-based chunking
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            # Simple chunking
            if len(content) <= chunk_size:
                return [content] if content.strip() else []
            chunks = []
            overlap = self.config['rag'].get('chunk_overlap', 200)
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks if chunks else [content]
        except Exception as e:
            print(f"Token-based chunking failed for {path}: {e}")
            return []

    def setup_watcher(self):
        """Setup file system watcher for auto-refresh"""
        if not WATCHDOG_AVAILABLE:
            return
        try:
            class Handler(FileSystemEventHandler):
                def __init__(self, rag):
                    self.rag = rag
                    self.last_refresh = time.time()
                def on_any_event(self, event):
                    refresh_interval = self.rag.config['rag']['index_refresh_interval']
                    if time.time() - self.last_refresh > refresh_interval:
                        print("File change detected. Refreshing index...")
                        self.rag.build_index()
                        self.last_refresh = time.time()
            observer = Observer()
            observer.schedule(Handler(self), '.', recursive=True)
            observer.start()
            print("File watcher started")
        except Exception as e:
            print(f"Failed to setup watcher: {e}")