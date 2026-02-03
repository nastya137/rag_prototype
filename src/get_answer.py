import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import time

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="style_manual_coursework_final_qualifying_work")
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
llm = OllamaLLM(model="llama3.2:latest", temperature=0.1)

# Шаблон запроса
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Ты эксперт в области оформления курсовых проектов и выпускных квалификационных работ. Дай ответ, учитывая ГОСТ по оформлению в области российского образования.

Documentation:
{context}

Question: {question}

Answer (уточняй, приводи конкретные цифры и значения, когда это возможно):"""
)

# Цепочка обработки вопросов
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
    context = "\n\n---SECTION---\n\n".join(documents)
    return context, documents

def get_llm_answer(question, context):
    answer = chain.invoke({"context": context[:2000], "question": question})
    return answer

def format_response(question, answer, source_chunks):
    response = f"**Question:** {question}\n\n"
    response += f"**Answer:** {answer}\n\n"
    response += "**Sources:**\n"
    for i, chunk in enumerate(source_chunks[:3], 1):
        preview = chunk[:100].replace("\n", " ") + "..."
        response += f"{i}. {preview}\n"
    return response

def enhanced_query_with_llm(question, n_results=5):
    context, documents = retrieve_context(question, n_results)
    answer = get_llm_answer(question, context)
    return format_response(question, answer, documents)

def stream_llm_answer(question, context):
    for chunk in chain.stream({"context": context[:2000], "question": question}):
        yield getattr(chunk, "content", str(chunk))


while(True):
    #Пример: "Допускается ли вписывать в текст выпускной квалификационной работы и курсового проекта отдельные слова, формулы и условные знаки?"
    question = input("\nВведите вопрос: ")
    context, documents = retrieve_context(question, n_results=3)
    print("Ответ: ", end="", flush=True)

    for token in stream_llm_answer(question, context):
        print(token, end="", flush=True)
        time.sleep(0.05)