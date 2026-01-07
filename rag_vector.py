import os
os.environ["HF_HUB_OFFLINE"] = "1"
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
class VectorRAG:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(
            self.config['rag']['embedding_model'],
            cache_folder=os.path.expanduser(self.config['paths']['embed_cache_dir'])
        )
        self.vectorizer = CountVectorizer()
        self.alpha = self.config['rag']['hybrid_alpha']
        self.index = {} # path: {'embs': np.array, 'bm25': sparse_matrix, 'chunks': list}
        self.contents = []
        self.paths = []
        self.build_index()
        if WATCHDOG_AVAILABLE and self.config['rag']['index_refresh_interval'] > 0:
            self.setup_watcher()

    # 在 VectorRAG 类中添加
    def rebuild_index(self):
        """Clear and rebuild the entire index"""
        print("Clearing existing RAG index...")
        self.index.clear()
        self.contents.clear()
        self.paths.clear()

        print("Rebuilding RAG index...")
        self.build_index()

    def build_index(self):
        """Build vector and BM25 index for all files"""
        print("Building RAG index...")
        self.paths = self.scan_files()
        self.contents = []
        chunks_per_file = []
        for path in self.paths:
            try:
                chunks = self.chunk_file(path, self.config['rag']['chunk_size'])
                if not chunks:
                    continue
                embs = self.model.encode(chunks)
                self.contents.extend(chunks)
                chunks_per_file.append(len(chunks))
                self.index[path] = {'embs': embs, 'chunks': chunks}
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
                print(f"Indexed {len(self.paths)} files with {len(self.contents)} chunks")
            except Exception as e:
                print(f"BM25 indexing failed: {e}")
    def scan_files(self):
        """Scan repository for indexable files"""
        exts = ['.md', '.markdown', '.txt', '.text', '.py', '.cpp', '.h', '.cc',
                '.c', '.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go']
        files = []
        try:
            for root, _, filenames in os.walk('.'):
                # 使用 os.path.normpath 规范化路径
                root_norm = os.path.normpath(root)
                # 跨平台检查隐藏目录
                skip_dirs = ['.git', '.agent', 'build', '__pycache__', 'cached_models']
                if any(skip_dir in root_norm.split(os.path.sep) for skip_dir in skip_dirs):
                    continue
                for file in filenames:
                    if any(file.endswith(ext) for ext in exts):
                        files.append(os.path.join(root, file))
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
            'python': '.py',
            'cpp': '.cpp',
            'c': '.c',
            'js': '.js',
            'java': '.java',
            'cs': '.cs',
            'go': '.go'
        }
        if lang:
            ext = lang_to_ext.get(lang, f'.{lang}')
            paths = [p for p in self.paths
                     if p.endswith(ext) or lang in os.path.basename(p).lower()]
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
            for path in paths:
                data = self.index.get(path, {})
                if not data:
                    continue
                # Semantic similarity
                embs = data.get('embs', [])
                if len(embs) > 0:
                    emb_scores = [cosine_similarity([query_emb], [emb])[0][0] for emb in embs]
                    avg_emb_score = np.mean(emb_scores)
                else:
                    avg_emb_score = 0
                # BM25 similarity
                bm25 = data.get('bm25')
                if bm25 is not None and bm25.shape[0] > 0:
                    bm25_scores = cosine_similarity(query_bm25, bm25)[0]
                    avg_bm25_score = np.mean(bm25_scores)
                else:
                    avg_bm25_score = 0
                # Hybrid score
                hybrid = self.alpha * avg_bm25_score + (1 - self.alpha) * avg_emb_score
                # Boost if query term in filename
                if query.lower() in os.path.basename(path).lower():
                    hybrid += 0.2
                scores[path] = hybrid
            # Rank and return
            ranked = sorted(scores, key=scores.get, reverse=True)[:top_k * 2]
            # Rerank if enabled
            if self.config['rag']['rerank_enabled'] and len(ranked) > top_k:
                ranked = self.rerank(ranked, query, mode)
            return ranked[:top_k]
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    def rerank(self, paths, query, mode):
        """LLM-based reranking"""
        try:
            snippets = []
            for path in paths:
                chunks = self.index[path].get('chunks', [])
                snippet = ' '.join(chunks[:3])[:500]
                snippets.append(snippet)
            rerank_prompt = self.config['tasks_optimizations'].get(
                mode, {}
            ).get('rerank_prompt', 'Rank by relevance: {query}').format(query=query)
            rerank_prompt += '\nSnippets:\n' + '\n'.join(
                [f"{i}: {s}" for i, s in enumerate(snippets)]
            )
            response = self.call_llm(rerank_prompt + '\nOutput ranked indices (space-separated):')
            # Parse indices
            indices = [int(x) for x in response.split() if x.isdigit()]
            reranked = [paths[i] for i in indices if i < len(paths)]
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
            headers = {}
            if self.config['llm'].get('api_key', ''):
                headers['Authorization'] = f'Bearer {self.config["llm"]["api_key"]}'
            resp = requests.post(
                self.config['llm']['endpoint'],
                json=payload,
                headers=headers,
                timeout=timeout
            )
            resp.raise_for_status()
            return resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.Timeout:
            print(f"LLM request timeout ({self.config.get('timeouts', {}).get('llm_request', 300)}s)")
            return ''
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ''
    def chunk_file(self, path, chunk_size):
        """Chunk file using tree-sitter or token-based fallback"""
        lang_map = {
            'py': 'python', 'cpp': 'cpp', 'c': 'c', 'js': 'javascript',
            'java': 'java', 'cs': 'c_sharp', 'go': 'go'
        }
        ext = path.split('.')[-1]
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
            # Try multiple encodings
            encodings = ['utf-8', 'gbk', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            # Last resort with error handling
            if content is None:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            # Simple chunking
            if len(content) <= chunk_size:
                return [content]
            chunks = []
            for i in range(0, len(content), chunk_size):
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
                    if time.time() - self.last_refresh > self.rag.config['rag']['index_refresh_interval']:
                        print("Refreshing index...")
                        self.rag.build_index()
                        self.last_refresh = time.time()
            observer = Observer()
            observer.schedule(Handler(self), '.', recursive=True)
            observer.start()
            print("File watcher started")
        except Exception as e:
            print(f"Failed to setup watcher: {e}")