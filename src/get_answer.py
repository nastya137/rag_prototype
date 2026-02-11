import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import get_model
import pathlib

root_project = pathlib.Path(__file__).absolute().parents[1]
client = chromadb.PersistentClient(path=root_project / "chroma_db")
model = get_model.Model.get_instance()
collection = client.get_collection(name="collection_1")
llm = OllamaLLM(model="llama3.2:latest", temperature=0.1)

# Шаблон запроса
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Ты эксперт в области оформления курсовых проектов и выпускных квалификационных работ. 
    Дай ответ, учитывая доступную тебе информацию из документа и, если указан, ГОСТ по оформлению в области российского образования.
    В первую очередь предоставляй информацию об оформлении текста документа, если в вопросе не указано иное.

Documentation:
{context}

Question: {question}

Answer (уточняй, приводи конкретные цифры и значения, когда это возможно):"""
)

chain = prompt_template | llm

def format_query_results(question, query_embedding, documents, metadatas):
    from sentence_transformers import util
    print(f"Question: {question}\n")
    for i, doc in enumerate(documents):
        doc_embedding = model.encode([doc])
        similarity = util.cos_sim(query_embedding, doc_embedding)[0][0].item()
        source = metadatas[i].get("document", "Unknown")
        print(f"Result {i+1} (similarity: {similarity:.3f}):")
        print(f"Document: {source}")
        print(f"Content: {doc[:300]}...")
        print()

def query_knowledge_base(question, n_results=5):
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    format_query_results(question, query_embedding, documents, metadatas)

def retrieve_context(question, n_results=5):
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })
    context = "\n\n---SECTION---\n\n".join(documents)
    return context, chunks

def get_llm_answer(question, context):
    answer = chain.invoke({"context": context[:2000], "question": question})
    return answer

def format_response(question, answer, source_chunks):
    response = f"{answer}\n\n"
    response += "Источники:\n"
    for i, chunk in enumerate(source_chunks, 1):
        preview = chunk["text"][:300].replace("\n", " ") + "..."
        distance = chunk["distance"]
        source = chunk["metadata"].get("document", "Unknown")
        response += (
            f"{i}. Источник: {source}\n"
            f"   Distance: {distance:.4f}\n"
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
