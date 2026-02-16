import networkx as nx
from pathlib import Path
import json
import chromadb
import elements_hardcode
from rules_extraction import extract_rules_from_chunk


def build_graph_from_chroma():
    root_project = Path(__file__).absolute().parents[1]
    client = chromadb.PersistentClient(path=root_project / "chroma_db")
    collection = client.get_collection(name="collection_1")

    G = nx.DiGraph()

    all_data = collection.get(include=["documents", "metadatas"])

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    for doc, meta in zip(documents, metadatas):
        chunk_id = f"chunk_{meta['doc_id']}_{meta['chunk_id']}"
        document_name = meta.get("document", "Unknown")
        G.add_node(
            chunk_id,
            type="Chunk",
            document=document_name,
        )
        rules = extract_rules_from_chunk(
            chunk_text=doc,
            chunk_id=chunk_id,
            elements=elements_hardcode.elements
        )

        for rule in rules:
            rule_id = rule["rule_id"]
            G.add_node(
                rule_id,
                type="Rule",
                text=rule["text"],
                rule_type=rule["type"]
            )
            G.add_edge(chunk_id, rule_id, type="contains_rule")
            for elem in rule["elements"]:
                elem_id = f"element_{elem}"

                G.add_node(
                    elem_id,
                    type="Element",
                    name=elem
                )

                G.add_edge(rule_id, elem_id, type="applies_to")

    return G


def save_graph(G, path="graph.json"):
    data = nx.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    G = build_graph_from_chroma()
    print(f"Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    root_project = Path(__file__).absolute().parents[1]
    save_graph(G, root_project/"graph.json")
    print("Граф сохранён в graph.json")
