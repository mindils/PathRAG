import os
import asyncio
from PathRAG import PathRAG, QueryParam
from PathRAG.llm_local import qwen_local_complete, qwen_local_embedding


async def query_documents():
    """Выполнение запросов к проиндексированным документам"""

    # Проверяем существование индекса
    if not os.path.exists("./pathrag_index"):
        print("Error: Index not found! Please run index_documents.py first.")
        return

    # Инициализируем PathRAG с теми же параметрами что и при индексации
    rag = PathRAG(
        working_dir="./pathrag_index",

        # Настройки моделей
        llm_model_func=qwen_local_complete,
        embedding_func=qwen_local_embedding,

        # Используем Qdrant
        # vector_storage="QdrantVectorDBStorage",

        # Настройки Qdrant
        # qdrant_config={
        #     "host": "localhost",
        #     "port": 6333,
        #     "grpc_port": 6334,
        # },

        # Те же параметры что и при индексации
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
        entity_summary_to_max_tokens=500,
        llm_model_max_token_size=32768,
        llm_model_max_async=4,
        embedding_batch_num=32,
        embedding_func_max_async=8,
        enable_llm_cache=True,

        addon_params={
            "language": "English",
            "entity_types": ["organization", "person", "technology", "location", "concept"],
        },

        log_level="INFO",
    )

    print("PathRAG Query Interface")
    print("=" * 50)
    print("Type 'exit' to quit")
    print("Type 'debug' before your query to see detailed context")
    print("=" * 50)

    while True:
        # Получаем запрос от пользователя
        query = input("\nEnter your query: ").strip()

        if query.lower() == 'exit':
            break

        if not query:
            continue

        # Проверяем режим отладки
        debug_mode = False
        if query.lower().startswith('debug '):
            debug_mode = True
            query = query[6:]  # Убираем 'debug '

        try:
            print("\nSearching for answer...")

            # Настройки запроса
            param = QueryParam(
                mode="hybrid",
                only_need_context=debug_mode,
                top_k=50,  # Больше узлов для поиска в документах
                max_token_for_text_unit=6000,  # Больше контекста
                max_token_for_global_context=4000,
                max_token_for_local_context=6000,

                # Специфичный тип ответа для нормативных документов
                response_type="""Структурированный ответ с указанием:
                1. Основные положения по запросу
                2. Ссылки на конкретные пункты документов
                3. Важные условия и ограничения
                4. Необходимые действия (если применимо)
                5. Сроки и суммы (если применимо)"""
            )

            # Выполняем запрос
            response = await rag.aquery(query, param)

            if debug_mode:
                print("\n--- DEBUG: Retrieved Context ---")
                print(response)
                print("--- END DEBUG ---\n")

                # Теперь получаем полный ответ
                param.only_need_context = False
                response = await rag.aquery(query, param)

            print("\n--- Answer ---")
            print(response)
            print("--- End Answer ---")

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("If the error persists, check that your models are running on the specified ports.")


async def test_queries():
    """Запуск тестовых запросов для оценки результатов"""

    # Проверяем существование индекса
    if not os.path.exists("./pathrag_index"):
        print("Error: Index not found! Please run index_documents.py first.")
        return

    # Инициализируем PathRAG
    rag = PathRAG(
        working_dir="./pathrag_index",
        llm_model_func=qwen_local_complete,
        embedding_func=qwen_local_embedding,
        vector_storage="QdrantVectorDBStorage",
        qdrant_config={
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
        },
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        log_level="WARNING",  # Меньше логов для тестов
    )

    # Тестовые запросы
    test_queries_corporate = [
        # Вопросы по оплате труда
        "Какой размер материальной помощи при отпуске для работника со стажем 2 года?",
        "Как рассчитывается доплата за работу в ночное время?",
        "Какие надбавки положены рабочим 5 разряда?",

        # Вопросы по компенсациям
        "Какой лимит компенсации проезда в 2025 году?",
        "Какие документы нужны для компенсации проезда детям?",
        "В какие сроки подавать документы на компенсацию?",

        # Вопросы по социальному пакету
        "Какой размер социального пакета для работника со стажем 4 года?",
        "Как оформить компенсацию за детский лагерь?",

        # Комплексные вопросы
        "Какие выплаты положены работнику филиала со стажем 3 года при уходе в отпуск?",
        "Перечислите все виды компенсаций для работников с детьми"
    ]

    print("Running test queries...")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}: {query}")
        print("-" * 40)

        try:
            param = QueryParam(mode="hybrid", top_k=30)
            response = await rag.aquery(query, param)
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 40)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Запуск тестовых запросов
        asyncio.run(test_queries())
    else:
        # Интерактивный режим
        asyncio.run(query_documents())