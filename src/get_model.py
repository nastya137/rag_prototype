from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

class Model:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            root_project = Path(__file__).absolute().parents[1]
            model_cache_dir = root_project / 'model'
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(model_cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(model_cache_dir)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            cls._instance = SentenceTransformer('multi-qa-mpnet-base-dot-v1', local_files_only=True)
        return cls._instance
