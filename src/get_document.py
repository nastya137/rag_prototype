from pathlib import Path
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
import os
import get_model
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
from get_qdrant_client import QdrantClientSingleton
import pdfplumber
import replicate
import io
from PIL import Image
import pymorphy3
import re

def call_dots_ocr(image: Image.Image) -> str:
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    output = replicate.run(
        "rednote-ai/dots-ocr:dots-ocr-1.5",
        input={
            "image": img_bytes,
            "prompt_mode": "prompt_layout_all_en"
        }
    )
    if isinstance(output, str):
        return output
    else:
        return ' '.join(output)

#решение для узких ячеек таблиц
def merge_split_words(text: str) -> str:
    morph = pymorphy3.MorphAnalyzer()
    words = text.split()
    if len(words) < 2:
        return text
    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            candidate = words[i] + words[i + 1]
            if morph.word_is_known(candidate):
                merged.append(candidate)
                i += 2
                continue
        merged.append(words[i])
        i += 1
    return ' '.join(merged)

#извлечение текста
def process_document(path):
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text_lines()
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            if not tables:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                continue
            all_items = []
            for line in lines:
                inside = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if (line['x0'] >= tx0 and line['x1'] <= tx1 and
                        line['top'] >= ty0 and line['bottom'] <= ty1):
                        inside = True
                        break
                if not inside:
                    all_items.append((line['top'], line['text']))
            for t in tables:
                cells = t.extract()
                if not cells:
                    continue
                cell_bboxes = t.cells if hasattr(t, 'cells') else None
                rows_text = []
                for i, row in enumerate(cells):
                    row_parts = []
                    for j, cell in enumerate(row):
                        if cell is None:
                            row_parts.append('')
                            continue
                        cell_clean = ' '.join(cell.split())
                        if ' ' not in cell_clean and len(cell_clean) > 15:
                            if cell_bboxes and i < len(cell_bboxes) and j < len(cell_bboxes[i]):
                                bbox_raw = cell_bboxes[i][j]
                                if isinstance(bbox_raw, dict):
                                    bbox = (bbox_raw.get('x0'), bbox_raw.get('top'),
                                            bbox_raw.get('x1'), bbox_raw.get('bottom'))
                                elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                                    bbox = tuple(bbox_raw)
                                else:
                                    bbox = None
                                if bbox and all(isinstance(v, (int, float)) for v in bbox):
                                    try:
                                        cell_img = page.within_bbox(bbox).to_image(resolution=150).original
                                        ocr_text = call_dots_ocr(cell_img)
                                        if ocr_text:
                                            cell_clean = ' '.join(ocr_text.split())
                                    except Exception as e:
                                        print(f"OCR failed for cell ({i},{j}): {e}")
                        row_parts.append(cell_clean)
                    row_text = ' '.join(row_parts).strip()
                    if row_text:
                        row_text = merge_split_words(row_text)
                        rows_text.append(row_text)
                y0 = t.bbox[1]
                for idx, row_text in enumerate(rows_text):
                    all_items.append((y0 + idx * 0.1, row_text))
            all_items.sort(key=lambda x: x[0])
            page_text = '\n'.join(text for _, text in all_items)
            pages_text.append(page_text)
    text = '\n'.join(pages_text)
    text = re.sub(r"([А-Яа-яЁёA-Za-z])\-\s*\n\s*([А-Яа-яЁёA-Za-z])", r"\1\2", text)
    return {"content": text, "source": path}


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
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def split_into_chunks(doc, text_splitter):
    doc_chunks = text_splitter.split_text(doc["content"])
    return [{"content": chunk, "source": doc["source"]} for chunk in doc_chunks]

allowed_ext = {".pdf", ".docx", ".txt"}

files_in_documents = [
    p for p in Path(output_folder).glob("*")
    if p.suffix.lower() in allowed_ext
]
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
embeddings = get_model.Model.encode_passages(documents)

print(f"Результат генерации эмбеддингов:")
print(f"  - Форма эмбеддинга: {embeddings.shape}")
print(f"  - Измерений векторов: {embeddings.shape[1]}")

#Построение базы знаний (Qdrant)
root_project = Path(__file__).absolute().parents[1]
client = QdrantClientSingleton.get_instance()

try:
    client.delete_collection("collection_1")
except:
    pass

collection_name = "collection_1"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

print(f"Создана коллекция: {collection_name}")

points = []

for idx, (vector, text, meta) in enumerate(zip(embeddings, documents, metadatas)):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "text": text,
                **meta,
                "node_type": "chunk"
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Всего в коллекции: {client.get_collection(collection_name).points_count}")

client.close()

