import json
import networkx as nx
import re
from pathlib import Path


def load_graph(path="graph.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower())


INTENT_KEYWORDS = {
    "font": ["шрифт"],
    "font_size": ["размер шрифта", "кегль", "пт"],
    "margins": ["поля", "мм"],
    "line_spacing": ["интервал", "межстроч"],
    "numbering": ["нумерац", "нумер"],
    "layout": ["располож", "размещ"],
    "formatting": ["оформлен", "оформля"],
}

ELEMENT_KEYWORDS = {
    "Иллюстрации": ["рисунк", "иллюстрац"],
    "Таблицы": ["таблиц"],
    "Формулы и уравнения": ["формул", "уравнен"],
    "Титульный лист": ["титуль"],
    "Содержание": ["содержан"],
    "Список использованных источников": ["литератур", "источник"],
    "Ссылки": ["ссылк"],
}


def detect_intent(question: str):
    q = normalize(question)
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in q for k in keys):
            return intent
    return None


def detect_element_from_question(question: str):
    q = normalize(question)
    for elem, keys in ELEMENT_KEYWORDS.items():
        if any(k in q for k in keys):
            return elem
    return None


def answer_from_graph(G, question: str):
    intent = detect_intent(question)
    element = detect_element_from_question(question)

    print(f"\nВопрос: {question}")
    print(f"intent: {intent}")
    print(f"element: {element}\n")

    rules = []

    for node, data in G.nodes(data=True):
        if data.get("type") != "Rule":
            continue

        if intent and data.get("rule_type") != intent:
            continue

        rules.append((node, data))

    if element:
        rules = [
            (rule_id, rule_data)
            for rule_id, rule_data in rules
            if any(
                G.nodes[nbr].get("name") == element
                for nbr in G.neighbors(rule_id)
                if G.edges[rule_id, nbr].get("type") == "applies_to"
            )
        ]

    if not rules:
        print("По графу ничего не найдено")
        return

    print("Найдены правила:\n")

    for i, (rule_id, rule_data) in enumerate(rules, 1):
        print(f"{i}. {rule_data['text']}")
        print(f"   Тип: {rule_data['rule_type']}")

        elems = [
            G.nodes[nbr]["name"]
            for nbr in G.neighbors(rule_id)
            if G.nodes[nbr].get("type") == "Element"
        ]

        print(f"   Элементы: {', '.join(elems) if elems else '—'}")

        chunks = [
            nbr for nbr in G.predecessors(rule_id)
            if G.nodes[nbr].get("type") == "Chunk"
        ]

        print("   Источники:")
        for ch in chunks:
            doc = G.nodes[ch].get("document")
            print(f"    - {doc} ({ch})")

        print("-" * 60)


if __name__ == "__main__":
    root_project = Path(__file__).absolute().parents[1]
    graph = load_graph(root_project / "graph.json")

    while True:
        q = input("\nВведите вопрос (или 'стоп'): ")
        if q.lower() == "стоп":
            break

        answer_from_graph(graph, q)
