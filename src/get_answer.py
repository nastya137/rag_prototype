import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from get_model import Model
import pathlib
from get_reranker import Reranker


root_project = pathlib.Path(__file__).absolute().parents[1]
reranker = Reranker.get_instance()
client = chromadb.PersistentClient(path=root_project / "chroma_db")
model = Model.get_instance()
collection = client.get_collection(name="collection_1")
llm = OllamaLLM(model="mistral", temperature=0.1)

# Шаблон запроса
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

def format_query_results(question, query_embedding, documents, metadatas):
    from sentence_transformers import util
    print(f"Question: {question}\n")
    doc_embeddings = Model.encode_passages(documents)
    for i, doc in enumerate(documents):
        similarity = util.cos_sim(query_embedding[0], doc_embeddings[i])[0].item()
        source = metadatas[i].get("document", "Unknown")
        print(f"Result {i+1} (similarity: {similarity:.3f}):")
        print(f"Document: {source}")
        print(f"Content: {doc[:300]}...")
        print()

def query_knowledge_base(question, n_results=5):
    query_embedding = Model.encode_query([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    format_query_results(question, query_embedding, documents, metadatas)

def retrieve_context(question, n_results=15, final_k=5, distance_threshold=0.4):
    query_embedding = Model.encode_query([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # print(f"\n--- Лучшие чанки до реранкинга ---\n")
    # for i, (doc, dist) in enumerate(zip(documents[:n_results], distances[:n_results])):
    #     print(f"{i+1}. {doc[:750]}...")
    #     print(f"Distance: {dist:.4f}")
    #     print("---")

    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
        if dist < distance_threshold
    ]

    if len(filtered) < final_k:
        filtered = list(zip(documents, metadatas, distances))

    filtered_docs = [item[0] for item in filtered]
    filtered_metas = [item[1] for item in filtered]
    filtered_dists = [item[2] for item in filtered]

    # Применяем реранкер
    pairs = [(question, doc) for doc in filtered_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(filtered_docs, filtered_metas, filtered_dists, scores),
        key=lambda x: x[3],
        reverse=True
    )

    top_chunks = ranked[:final_k]
    context_docs = [item[0] for item in top_chunks]

    chunks = []
    for doc, meta, dist, score in top_chunks:
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": float(dist),
            "score": float(score)
        })

    context = "\n\n---SECTION---\n\n".join(context_docs)

    # print(f"\n--- Лучшие чанки после реранкинга ---\n")
    # for i, chunk in enumerate(chunks):
    #     print(f"{i+1}. {chunk['text'][:750]}...")
    #     print(f"Distance: {chunk['distance']:.4f}")
    #     print(f"Score: {chunk['score']:.4f}")
    #     print("---")

    return context, chunks

def get_llm_answer(question, context):
    answer = chain.invoke({"context": context[:1000], "question": question})
    return answer

def format_response(question, answer, source_chunks):
    response = f"{answer}\n\n"
    response += "Источники:\n"
    for i, chunk in enumerate(source_chunks, 1):
        preview = chunk["text"][:300].replace("\n", " ") + "..."
        score = chunk["score"]
        source = chunk["metadata"].get("document", "Unknown")
        response += (
            f"{i}. Источник: {source}\n"
            f"   Distance (Chroma): {chunk['distance']:.4f}\n"
            f"   Score: {score:.4f}\n"
            f"   Текст: {preview}\n\n"
        )
    return response

def enhanced_query_with_llm(question, n_results=5):
    context, chunks = retrieve_context(question, n_results)
    answer = get_llm_answer(question, context)
    return format_response(question, answer, chunks)


def stream_llm_answer(question, context):
    for chunk in chain.stream({"context": context[:2000], "question": question}):
        yield getattr(chunk, "content", str(chunk))

question = ""

while(True):
    #Пример: "Допускается ли вписывать в текст выпускной квалификационной работы и курсового проекта отдельные слова, формулы и условные знаки?"
    question = input("\nВведите вопрос: ")
    if (question=="стоп"):
        break
    enhanced_response = enhanced_query_with_llm(question)
    print(enhanced_response)
