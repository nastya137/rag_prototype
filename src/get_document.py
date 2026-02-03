from pathlib import Path
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
import os
from sentence_transformers import SentenceTransformer
import chromadb

#Получение пути к документу
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
filename = "metod_ukazaniya_po_oformleniyu_vkr_0.docx"
output_folder = os.path.join(parent_dir, 'documents')
file_path = Path(output_folder) / filename

#Обработка документа для чанкирования
md = MarkItDown()
result = md.convert(file_path)
content = result.text_content

processed_document = {
    'source': file_path,
    'content': content
}

documents = [processed_document]
print(f"Document ready: {len(processed_document['content']):,} characters")

#Чанкирование
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=90,
    separators=["\n\n", "\n", ". ", " ", ""]
)
def process_document(doc, text_splitter):
    doc_chunks = text_splitter.split_text(doc["content"])
    return [{"content": chunk, "source": doc["source"]} for chunk in doc_chunks]

all_chunks = []
for doc in documents:
    doc_chunks = process_document(doc, text_splitter)
    all_chunks.extend(doc_chunks)

source_counts = Counter(chunk["source"] for chunk in all_chunks)
chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

print(f"Total chunks created: {len(all_chunks)}")
print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")
print(f"Source document: {Path(documents[0]['source']).name}")


# Получение эмбеддингов
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

documents = [chunk["content"] for chunk in all_chunks]
embeddings = model.encode(documents)

print(f"Embedding generation results:")
print(f"  - Embeddings shape: {embeddings.shape}")
print(f"  - Vector dimensions: {embeddings.shape[1]}")

#Построение базы знаний (ChromaDB)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="style_manual_coursework_final_qualifying_work",
    metadata={"description": "Style manual for coursework and final qualifying work(KubSU, Faculty of Physics and Technology)"}
)

print(f"Created collection: {collection.name}")
print(f"Collection ID: {collection.id}")

metadatas = [{"document": Path(chunk["source"]).name} for chunk in all_chunks]

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))],
)

print(f"Collection count: {collection.count()}")
