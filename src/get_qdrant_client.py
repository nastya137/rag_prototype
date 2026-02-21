from qdrant_client import QdrantClient
from pathlib import Path

class QdrantClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            root_project = Path(__file__).absolute().parents[1]
            cls._instance = QdrantClient(path=str(root_project / "qdrant_db"))
        return cls._instance
