# rag.py - Wrapper for rag_vector.py to maintain compatibility
from rag_vector import VectorRAG


class RAGIndexer:
    """Wrapper class for RAG indexing operations"""

    def __init__(self, config, workspace_dir=None):
        self.config = config
        self.workspace_dir = workspace_dir or '.'
        self.rag = VectorRAG(config)

    def rebuild(self):
        """Rebuild the RAG index"""
        self.rag.rebuild_index()
        return "RAG index rebuilt successfully"

    def search(self, query, top_k=5, mode='code', lang=None):
        """Search the RAG index"""
        return self.rag.search(query, top_k, mode, lang)