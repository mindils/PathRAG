import os
import glob
from PathRAG import PathRAG, QueryParam
from PathRAG.llm_local import qwen_local_complete, qwen_local_embedding
import asyncio



async def index_documents():
    """Индексация документов из папки input"""

    # Настройка PathRAG с вашим стеком
    rag = PathRAG(
        working_dir="./pathrag_index",

        # Настройки моделей
        llm_model_func=qwen_local_complete,
        embedding_func=qwen_local_embedding,

        # Используем Qdrant для векторного хранилища
        # vector_storage="QdrantVectorDBStorage",

        # Настройки Qdrant
        # qdrant_config={
        #     "host": "localhost",
        #     "port": 6333,
        #     "grpc_port": 6334,
        # },

        # Параметры чанков
        chunk_token_size=1200,
        chunk_overlap_token_size=100,

        # Параметры извлечения сущностей
        entity_extract_max_gleaning=1,  # Количество итераций извлечения
        entity_summary_to_max_tokens=500,

        # Параметры моделей
        llm_model_max_token_size=32768,
        llm_model_max_async=4,  # Количество параллельных запросов к LLM
        embedding_batch_num=32,
        embedding_func_max_async=8,

        # Включаем кэширование
        enable_llm_cache=True,

        # Дополнительные параметры
        addon_params={
            "language": "English",  # Или "Russian" если документы на русском
            "entity_types": ["organization", "person", "technology", "location", "concept"],
        },

        # Уровень логирования
        log_level="INFO",
    )

    # Получаем список файлов для индексации
    input_files = []

    # Поддерживаем разные форматы
    for pattern in ["*.txt", "*.md", "*.pdf"]:
        input_files.extend(glob.glob(os.path.join("input", pattern)))

    if not input_files:
        print("No files found in input directory!")
        return

    print(f"Found {len(input_files)} files to index")

    # Читаем и индексируем файлы
    documents = []
    for file_path in input_files:
        print(f"Reading {file_path}...")

        if file_path.endswith('.pdf'):
            # Для PDF нужна дополнительная библиотека
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                documents.append(content)
            except ImportError:
                print("PyPDF2 not installed. Skipping PDF files. Install with: pip install PyPDF2")
                continue
        else:
            # Для текстовых файлов
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)

    if documents:
        print(f"\nStarting indexing of {len(documents)} documents...")
        print("This may take a while depending on document size and model speed...")

        # Индексируем документы
        await rag.ainsert(documents)

        print("\nIndexing completed successfully!")
        print(f"Index saved to: {rag.working_dir}")
    else:
        print("No documents to index!")


if __name__ == "__main__":
    # Создаем папку input если её нет
    os.makedirs("input", exist_ok=True)

    print("PathRAG Document Indexer")
    print("=" * 50)
    print("Place your documents in the 'input' folder")
    print("Supported formats: .txt, .md, .pdf")
    print("=" * 50)

    # Запускаем индексацию
    asyncio.run(index_documents())