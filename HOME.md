Изучив код PathRAG, я помогу вам настроить его для вашего стека. PathRAG - это улучшенный подход к RAG, который строит граф знаний из документов и использует релевантные пути из этого графа для генерации ответов.

## Суть PathRAG

PathRAG работает в несколько этапов:
1. **Разбивка документов на чанки** (chunks)
2. **Извлечение сущностей и связей** из каждого чанка с помощью LLM
3. **Построение графа знаний** где узлы - сущности, а рёбра - связи
4. **При запросе**: извлечение ключевых слов высокого/низкого уровня → поиск релевантных путей в графе → генерация ответа

## Пошаговая реализация

### Шаг 1: Установка зависимостей

```bash
cd PathRAG
pip install -e .
pip install qdrant-client
```

### Шаг 2: Создание функций для работы с локальными моделями

Создайте файл `PathRAG/llm_local.py`:### Шаг 3: Создание хранилища Qdrant

Создайте файл `PathRAG/storage_qdrant.py`:### Шаг 4: Обновление PathRAG для поддержки новых компонентов

Добавьте в файл `PathRAG/PathRAG.py` после других импортов lazy_external_import:

```python
# В секцию импортов после других lazy_external_import
QdrantVectorDBStorage = lazy_external_import(".storage_qdrant", "QdrantVectorDBStorage")

# В метод _get_storage_class() добавьте:
"QdrantVectorDBStorage": QdrantVectorDBStorage,
```

### Шаг 5: Создание скрипта для индексации документов

Создайте файл `index_documents.py`:### Шаг 6: Создание скрипта для запросов

Создайте файл `query_documents.py`:### Шаг 7: Скрипт проверки окружения

Создайте файл `check_setup.py`:### Шаг 8: Обновление импортов

Добавьте в `PathRAG/__init__.py`:

```python
from .PathRAG import PathRAG as PathRAG, QueryParam as QueryParam
from .llm_local import qwen_local_complete, qwen_local_embedding
from .storage_qdrant import QdrantVectorDBStorage
```

## Полная инструкция по запуску

### 1. Установка и настройка

```bash
# Клонируйте или скопируйте PathRAG
cd PathRAG

# Установите зависимости
pip install -e .
pip install qdrant-client aiohttp PyPDF2

# Запустите Qdrant (если не запущен)
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant

# Проверьте настройку
python check_setup.py
```

### 2. Подготовка моделей

Убедитесь что ваши модели запущены:
- LLM на `localhost:8801`
- Embedding на `localhost:8802`

### 3. Индексация документов

```bash
# Создайте папку input и поместите туда документы
mkdir input
# Скопируйте ваши .txt, .md, .pdf файлы в папку input

# Запустите индексацию
python index_documents.py
```

### 4. Выполнение запросов

```bash
# Интерактивный режим
python query_documents.py

# Или тестовые запросы
python query_documents.py test
```

## Основные изменения в коде для вашего стека

1. **Новые функции LLM** (`llm_local.py`):
   - `qwen_local_complete` - для работы с вашей LLM
   - `qwen_local_embedding` - для работы с embedding моделью

2. **Поддержка Qdrant** (`storage_qdrant.py`):
   - `QdrantVectorDBStorage` - класс для хранения векторов в Qdrant

3. **Конфигурация PathRAG**:
   - Использование `vector_storage="QdrantVectorDBStorage"`
   - Настройка подключения через `qdrant_config`
   - Указание кастомных функций для LLM и embeddings

## Особенности PathRAG

PathRAG улучшает обычный RAG следующим образом:

1. **Построение графа знаний**: Извлекает сущности и связи между ними
2. **Двухуровневый поиск**: 
   - Высокий уровень - концепции и темы
   - Низкий уровень - конкретные детали
3. **Поиск путей**: Находит релевантные пути между сущностями в графе
4. **Уменьшение избыточности**: Использует только нужные пути, а не все связанные документы

Это позволяет получать более точные и логически связанные ответы.