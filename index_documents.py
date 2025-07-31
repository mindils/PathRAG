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

        # Параметры чанков - увеличиваем для нормативных документов
        chunk_token_size=2000,  # Увеличено для захвата полных разделов
        chunk_overlap_token_size=200,  # Больше перекрытие для связности

        # Параметры извлечения сущностей
        entity_extract_max_gleaning=2,  # Больше итераций для полноты
        entity_summary_to_max_tokens=1000,  # Больше токенов для описаний

        # Дополнительные параметры
        addon_params={
            "language": "Russian",
            "entity_types": [
                "organization",  # АА "Компания", филиалы, департаменты
                "person",  # работники, руководители
                "document",  # приказы, положения, заявления
                "regulation",  # нормативные акты, положения
                "department",  # структурные подразделения
                "position",  # должности
                "process",  # процессы и процедуры
                "term",  # термины и определения
                "date",  # даты и сроки
                "amount",  # суммы, размеры, проценты
                "period",  # периоды времени
                "condition"  # условия и требования
            ],
            "example_number": 2,  # Использовать примеры

            # Дополнительные параметры для русского языка
            "extract_patterns": {
                "document_refs": r"(?:приказ|положение|регламент|инструкция).*?№\s*\d+.*?от\s*\d{1,2}\.\d{1,2}\.\d{4}",
                "amounts": r"\d+(?:\s*\d{3})*(?:\.\d{2})?\s*(?:руб(?:лей|ля|ль)?|%|процент)",
                "dates": r"\d{1,2}\.\d{1,2}\.\d{4}",
                "periods": r"\d+\s*(?:лет|года?|месяц|месяца|месяцев|дня|дней|календарных дней)"
            }
        },

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