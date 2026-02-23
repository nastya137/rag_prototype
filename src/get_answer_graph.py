from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from get_model import Model
from get_reranker import Reranker
import re
import os
from dotenv import load_dotenv
from entity_rules import ENTITY_RULES

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

reranker = Reranker.get_instance()
model = Model.get_instance()
llm = OllamaLLM(model="mistral", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Ты эксперт в области оформления курсовых проектов и выпускных квалификационных работ. 
Дай ответ, учитывая доступную тебе информацию из документа и, если указан, ГОСТ по оформлению в области российского образования.
В первую очередь предоставляй информацию об оформлении документа, если в вопросе не указано иное. Если информации не хватает, не придумывай её и не добавляй ту информацию, которая не связана с вопросом.

Documentation:
{context}

Question: {question}

Answer (уточняй, приводи конкретные цифры и значения, когда это возможно):"""
)

chain = prompt_template | llm





def extract_entities_from_question(question: str):
    q = question.lower()
    found = []

    for entity, patterns in ENTITY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, q):
                found.append(entity)
                break

    return list(set(found))


def retrieve_context_from_graph(question, final_k=5):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    entities = extract_entities_from_question(question)

    if not entities:
        return "", []

    query = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE e.name IN $entities
    RETURN c.text AS text, e.name AS entity, c.id AS chunk_id
    LIMIT 50
    """

    chunks = []

    with driver.session() as session:
        results = session.run(query, entities=entities)
        for r in results:
            chunks.append({
                "text": r["text"],
                "metadata": {
                    "entity": r["entity"],
                    "chunk_id": r["chunk_id"]
                }
            })

    driver.close()

    if not chunks:
        return "", []

    docs = [c["text"] for c in chunks]

    pairs = [(question, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )[:final_k]

    context_docs = [item[0]["text"] for item in ranked]

    final_chunks = []
    for item, score in ranked:
        final_chunks.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": float(score)
        })

    context = "\n\n---SECTION---\n\n".join(context_docs)

    return context, final_chunks


def get_llm_answer(question, context):
    return chain.invoke({"context": context[:1500], "question": question})


def format_response(answer, source_chunks):
    response = f"{answer}\n\nИсточники (граф знаний):\n"
    for i, chunk in enumerate(source_chunks, 1):
        preview = chunk["text"][:300].replace("\n", " ") + "..."
        entity = chunk["metadata"].get("entity", "Unknown")
        response += (
            f"{i}. Сущность: {entity}\n"
            f"   Score: {chunk['score']:.4f}\n"
            f"   Текст: {preview}\n\n"
        )
    return response


def enhanced_query_with_llm(question):
    context, chunks = retrieve_context_from_graph(question)
    answer = get_llm_answer(question, context)
    return format_response(answer, chunks)


while True:
    question = input("\nВведите вопрос: ")
    if question.lower() == "стоп":
        break

    try:
        enhanced_response = enhanced_query_with_llm(question)
        print(enhanced_response)
    except Exception as e:
        print("Ошибка при выполнении запроса:", e)
