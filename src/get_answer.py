from get_qdrant_client import QdrantClientSingleton
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from get_model import Model
import pathlib
from get_reranker import Reranker

root_project = pathlib.Path(__file__).absolute().parents[1]
reranker = Reranker.get_instance()
client = QdrantClientSingleton.get_instance()
collection_name = "collection_1"
model = Model.get_instance()
llm = OllamaLLM(model="mistral", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""### Роль ###
Ты — ассистент на базе внутренней базы знаний. Твоя задача — помочь инженеру проанализировать инцидент, найдя релевантную информацию в документации и прошлых решениях.

### Источник данных ###
Для ответа используй ИСКЛЮЧИТЕЛЬНО предоставленные ниже материалы из базы знаний:
{context}

### Строгие инструкции ###

1.  Анализ запроса: Разбери входящее сообщение на компоненты:
    *   Система/сервис 
    *   Компонент/модуль
    *   Номер заявки/идентификатор
    *   Текст ошибки/проблемы

2.  Основа ответа: Каждое утверждение в ответе должно иметь прямое подтверждение в предоставленном контексте. Запрещено:
    *   Придумывать команды, скрипты или пути
    *   Расшифровывать аббревиатуры без их явного объяснения в контексте
    *   Предполагать работу систем, не описанных в контексте

3.  Структура ответа:

    Интерпретация инцидента:
    На основе запроса выявлены ключевые элементы: [перечисли элементы из п.1]. 
    В контексте найдена следующая релевантная информация: [кратко опиши, что именно в контексте относится к этим элементам].

    Рекомендуемые проверки из базы знаний:
    [Строго на основе контекста предложи последовательность проверок. Если в контексте есть:
    - Конкретные команды → укажи их
    - Названия инструментов → укажи их
    - Пути к логам → укажи их
    - Процедуры проверки → опиши их
    Если такой информации нет, не придумывай!]

    Возможные решения из базы знаний:
    [Если в контексте есть описание решения подобных проблем, перечисли их строго по материалам. Если нет, так и укажи.]

    Для эскалации:
    Если рекомендации не помогли или информация в базе знаний недостаточна:
    - Убедись, что выполнены все проверки из статей [перечисли номера статей или названия из контекста]
    - Подготовь данные для передачи в L3: [укажи, какие данные следует собрать согласно контексту]

### Важно: ###
- Если в контексте нет информации по какой-либо части запроса, прямо укажи это
- Используй терминологию точно в том виде, в котором она представлена в контексте
- Не добавляй интерпретации, не основанные на контексте

### Вопрос инженера ###
{question}"""
)

chain = prompt_template | llm


def retrieve_context(question, n_results=15, final_k=5, similarity_threshold=0.3):
    query_embedding = Model.encode_query([question])[0].tolist()

    results = client._client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=n_results
    )

    documents = [hit.payload["text"] for hit in results]
    metadatas = [hit.payload for hit in results]
    similarities = [hit.score for hit in results]

    filtered = [
        (doc, meta, sim)
        for doc, meta, sim in zip(documents, metadatas, similarities)
        if sim >= similarity_threshold
    ]

    if len(filtered) < final_k:
        filtered = list(zip(documents, metadatas, similarities))

    filtered_docs = [item[0] for item in filtered]
    filtered_metas = [item[1] for item in filtered]
    filtered_sims = [item[2] for item in filtered]

    pairs = [(question, doc) for doc in filtered_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(filtered_docs, filtered_metas, filtered_sims, scores),
        key=lambda x: x[3],
        reverse=True
    )

    top_chunks = ranked[:final_k]

    context_docs = [item[0] for item in top_chunks]

    chunks = []
    for doc, meta, sim, score in top_chunks:
        chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity": float(sim),
            "score": float(score)
        })

    context = "\n\n---SECTION---\n\n".join(context_docs)

    return context, chunks


def get_llm_answer(question, context):
    return chain.invoke({"context": context[:1500], "question": question})


def format_response(question, answer, source_chunks):
    response = f"{answer}\n\nИсточники:\n"
    for i, chunk in enumerate(source_chunks, 1):
        preview = chunk["text"][:300].replace("\n", " ") + "..."
        source = chunk["metadata"].get("document", "Unknown")
        response += (
            f"{i}. Источник: {source}\n"
            f"   Similarity (Qdrant): {chunk['similarity']:.4f}\n"
            f"   Reranker score: {chunk['score']:.4f}\n"
            f"   Текст: {preview}\n\n"
        )
    return response


def enhanced_query_with_llm(question, n_results=5):
    context, chunks = retrieve_context(question, n_results=n_results)
    answer = get_llm_answer(question, context)
    return format_response(question, answer, chunks)


while True:
    question = input("\nВведите вопрос: ")
    if question.lower() == "стоп":
        client.close()
        break

    try:
        enhanced_response = enhanced_query_with_llm(question)
        print(enhanced_response)
    except Exception as e:
        print("Ошибка при выполнении запроса:", e)

