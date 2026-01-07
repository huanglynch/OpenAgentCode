# test_rag_cache.py
import os
import yaml

# 模拟配置
config = {
    '_original_cwd': os.getcwd(),
    'paths': {
        'embed_cache_dir': 'cached_models/'
    },
    'rag': {
        'embedding_model': 'all-mpnet-base-v2',
        'hybrid_alpha': 0.4,
        'chunk_size': 384,
        'chunk_overlap': 200,
        'index_refresh_interval': 0
    }
}

# 测试缓存路径解析
from rag_vector import VectorRAG

print("Testing RAG cache path handling...")
print(f"Original CWD: {config['_original_cwd']}")

try:
    rag = VectorRAG(config)
    print("✓ RAG initialized successfully")
    print(f"Model loaded: {rag.model}")
except Exception as e:
    print(f"✗ Failed: {e}")