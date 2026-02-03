import get_document

def format_query_results(question, query_embedding, documents, metadatas):
    from sentence_transformers import util

    print(f"Question: {question}\n")

    for i, doc in enumerate(documents):
        # Calculate accurate similarity using sentence-transformers util
        doc_embedding =get_document. model.encode([doc])
        similarity = util.cos_sim(query_embedding, doc_embedding)[0][0].item()
        source = metadatas[i].get("document", "Unknown")

        print(f"Result {i+1} (similarity: {similarity:.3f}):")
        print(f"Document: {source}")
        print(f"Content: {doc[:300]}...")
        print()


def query_knowledge_base(question, n_results=5):
    query_embedding =get_document.model.encode([question])

    results =get_document.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Extract results and format them
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    format_query_results(question, query_embedding, documents, metadatas)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Initialize the local LLM
llm = OllamaLLM(model="llama3.2:latest", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Ты эксперт в области оформления курсовых проектов и выпускных квалификационных работ. Дай ответ, учитывая ГОСТ по оформлению в области российского образования.

Documentation:
{context}

Question: {question}

Answer (уточняй, приводи примеры и конкретные цифры и значения, когда это возможно):"""
)
chain = prompt_template | llm

def retrieve_context(question, n_results=5):
    """Retrieve relevant context using embeddings"""
    query_embedding =get_document.model.encode([question])
    results = get_document.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    context = "\n\n---SECTION---\n\n".join(documents)
    return context, documents


def get_llm_answer(question, context):
    answer = chain.invoke(
        {
            "context": context[:2000],
            "question": question,
        }
    )
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
    for chunk in chain.stream({
        "context": context[:2000],
        "question": question,
    }):
        yield getattr(chunk, "content", str(chunk))

import time

# Test the streaming functionality
question = "Допускается ли Вписывать в текст выпускной квалификационной работы и курсового проекта отдельные слова, формулы и условные знаки?"
context, documents = retrieve_context(question, n_results=3)

print("Question:", question)
print("Answer: ", end="", flush=True)

# Stream the answer token by token
for token in stream_llm_answer(question, context):
    print(token, end="", flush=True)
    time.sleep(0.05)  # Simulate real-time typing effect