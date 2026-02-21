from neo4j import GraphDatabase
from pathlib import Path
from typing import List, Dict
from get_qdrant_client import QdrantClientSingleton
import os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class GraphClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, params)

KEYWORDS = [
    "ГОСТ", "таблица", "рисунок", "шрифт",
    "межстрочный интервал", "поля", "нумерация", "список литературы",
    "приложение", "оформление", "введение", "основная часть",
    "заключение", "иллюстрации", "размер шрифта", "поля",
    "титульный лист", "реферат", "список использованных источников"
]

def extract_entities(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for kw in KEYWORDS:
        if kw.lower() in lower:
            found.append(kw)
    return list(set(found))

def init_schema(graph: GraphClient):
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

def save_chunk(
    graph: GraphClient,
    doc_name: str,
    chunk_id: str,
    text: str,
    entities: List[str]
):
    graph.run("""
    MERGE (d:Document {name: $doc_name})
    MERGE (c:Chunk {id: $chunk_id})
    SET c.text = $text
    MERGE (d)-[:HAS_CHUNK]->(c)
    """, doc_name=doc_name, chunk_id=chunk_id, text=text)

    for ent in entities:
        graph.run("""
        MERGE (e:Entity {name: $entity})
        WITH e
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (c)-[:MENTIONS]->(e)
        """, entity=ent, chunk_id=chunk_id)

def build_graph_from_chunks(chunks: List[Dict]):
    graph = GraphClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    init_schema(graph)

    for chunk in chunks:
        doc_name = chunk["metadata"]["document"]
        chunk_id = f'{chunk["metadata"]["doc_id"]}_{chunk["metadata"]["chunk_id"]}'
        text = chunk["text"]

        entities = extract_entities(text)

        save_chunk(graph, doc_name, chunk_id, text, entities)

    graph.close()

def load_chunks_from_qdrant(collection_name: str):
    root_project = Path(__file__).absolute().parents[1]
    client = QdrantClientSingleton.get_instance()

    chunks = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )

        if not points:
            break

        for p in points:
            chunks.append({
                "text": p.payload.get("text", ""),
                "metadata": {
                    "document": p.payload.get("document"),
                    "doc_id": p.payload.get("doc_id"),
                    "chunk_id": p.payload.get("chunk_id")
                }
            })

        if offset is None:
            break

    client.close()
    return chunks

if __name__ == "__main__":
    print("Построение графа знаний...")

    chunks = load_chunks_from_qdrant("collection_1")
    print(f"Загружено чанков из Qdrant: {len(chunks)}")

    build_graph_from_chunks(chunks)
    print("Граф построен.")