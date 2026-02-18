from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
model_name = "intfloat/multilingual-e5-base"
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
                    model_name,
                    cache_folder=str(model_cache_dir),
                    local_files_only=True)
                print("Модель загружена из локального кэша.")
            except OSError:
                try:
                    print("Локально модель не найдена. Загрузка из Hugging Face...")
                    cls._instance = SentenceTransformer(
                        model_name,
                        cache_folder=str(model_cache_dir),
                        local_files_only=False,
                    )
                except Exception as e:
                    print("Ошибка загрузки модели:", e)
                    raise RuntimeError("Embedding model initialization failed")

                print("Модель загружена и сохранена локально.")
        return cls._instance

    @classmethod
    def encode_query(cls, texts):
        model = cls.get_instance()
        texts = [f"query: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True)

    @classmethod
    def encode_passages(cls, texts):
        model = cls.get_instance()
        texts = [f"passage: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True)
