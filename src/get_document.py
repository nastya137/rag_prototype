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
    chunk_size=450,
    chunk_overlap=90,
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
        print(f"Processing document: {file_path} ...")
        proc_doc = process_document(file_path)
        print(f"Document processed: {proc_doc['source'].name}")
        all_processed_docs.append(proc_doc)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

all_chunks = []
previous_chunks_num = 0
for doc in all_processed_docs:
    doc_chunks = split_into_chunks(doc, text_splitter)
    all_chunks.extend(doc_chunks)

source_counts = Counter(chunk["source"] for chunk in all_chunks)
chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

print(f"Total chunks created: {len(all_chunks)}")
print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")

documents = [chunk["content"] for chunk in all_chunks]

# Получение эмбеддингов
model = get_model.Model.get_instance()

documents = [chunk["content"] for chunk in all_chunks]
embeddings = model.encode(documents)

print(f"Embedding generation results:")
print(f"  - Embeddings shape: {embeddings.shape}")
print(f"  - Vector dimensions: {embeddings.shape[1]}")

#Построение базы знаний (ChromaDB)
root_project = Path(__file__).absolute().parents[1]
client = chromadb.PersistentClient(path=root_project / "chroma_db")

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
