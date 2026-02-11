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
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(model_cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(model_cache_dir)
            try:
                cls._instance = SentenceTransformer(
                    'multi-qa-mpnet-base-dot-v1',
                    cache_folder=str(model_cache_dir),
                    local_files_only=True)
                print("Модель загружена из локального кэша.")
            except OSError:
                print("Локально модель не найдена. Загрузка из Hugging Face...")
                cls._instance = SentenceTransformer(
                    'multi-qa-mpnet-base-dot-v1',
                    cache_folder=str(model_cache_dir),
                    local_files_only=False,
                )

                print("Модель загружена и сохранена локально.")
        return cls._instance
