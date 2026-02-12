from pathlib import Path
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
import os
import get_model
import chromadb

#Получение пути к документам
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model = get_model.Model.get_instance()
output_folder = os.path.join(parent_dir, 'documents')
files_in_documents = list(Path(output_folder).glob('*'))

# Чанкирование и подготовка данных
md = MarkItDown()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def process_document(file_path):
    result = md.convert(file_path)
    content = result.text_content
    processed_doc = {'source': file_path, 'content': content}
    return processed_doc

def split_into_chunks(doc, text_splitter):
    doc_chunks = text_splitter.split_text(doc["content"])
    return [{"content": chunk, "source": doc["source"]} for chunk in doc_chunks]

all_processed_docs = []
for file_path in files_in_documents:
    try:
        print(f"Обработка документа: {file_path} ...")
        proc_doc = process_document(file_path)
        print(f"Документ обработан: {proc_doc['source'].name}")
        all_processed_docs.append(proc_doc)
    except Exception as e:
        print(f"Ошибка обработки: {file_path}: {e}")

all_chunks = []
previous_chunks_num = 0
for doc in all_processed_docs:
    doc_chunks = split_into_chunks(doc, text_splitter)
    all_chunks.extend(doc_chunks)

source_counts = Counter(chunk["source"] for chunk in all_chunks)
chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

print(f"Всего создано чанков: {len(all_chunks)}")
print(f"Размер чанков: {min(chunk_lengths)}-{max(chunk_lengths)} символов")

documents = []
metadatas = []
ids = []

for doc_id, doc in enumerate(all_processed_docs):
    doc_chunks = split_into_chunks(doc, text_splitter)

    for chunk_id, chunk in enumerate(doc_chunks):
        documents.append(chunk["content"])

        metadatas.append({
            "document": Path(chunk["source"]).name,
            "doc_id": doc_id,
            "chunk_id": chunk_id
        })

        ids.append(f"{doc_id}_{chunk_id}")

# Получение эмбеддингов
print(f"Генерация эмбеддингов...")
model = get_model.Model.get_instance()
embeddings = model.encode(documents)

print(f"Результат генерации эмбеддингов:")
print(f"  - Форма эмбеддинга: {embeddings.shape}")
print(f"  - Измерений векторов: {embeddings.shape[1]}")

#Построение базы знаний (ChromaDB)
root_project = Path(__file__).absolute().parents[1]
client = chromadb.PersistentClient(path=root_project / "chroma_db")

collection = client.get_or_create_collection(
    name="collection_1",
    metadata={"description": "Тестовая коллекция"}
)

if collection.count() > 0:
    print("Коллекция уже содержит данные. Пересоздание...")
    client.delete_collection("collection_1")
    collection = client.get_or_create_collection(
        name="collection_1",
        metadata={"description": "Тестовая коллекция"}
    )

print(f"Создана коллекция: {collection.name}")
print(f"ID коллекции: {collection.id}")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print(f"Collection count: {collection.count()}")
