from sentence_transformers import CrossEncoder
import os
from pathlib import Path

class Reranker:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            root_project = Path(__file__).absolute().parents[1]
            reranker_cache_dir = root_project / 'model'
            reranker_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(reranker_cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(reranker_cache_dir)
            try:
                cls._instance = CrossEncoder(
                    "cross-encoder/mmarco-mMiniLMv2-L6-H384-v1",
                    cache_folder=str(root_project / 'model'),
                    local_files_only=True
                )
                print("Реранкер загружен из локального кэша.")
            except OSError:
                try:
                    print("Локально реранкер не найден. Загрузка...")
                    cls._instance = CrossEncoder(
                        "cross-encoder/mmarco-mMiniLMv2-L6-H384-v1",
                        cache_folder=str(root_project / 'model'),
                        local_files_only=False
                    )
                except Exception as e:
                    print("Не удалось загрузить реранкер:", e)
                    raise RuntimeError("Reranker initialization failed")

                print("Реранкер загружен и сохранен локально.")
        return cls._instance