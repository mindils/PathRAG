import asyncio
from dataclasses import dataclass
from typing import Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.asyncio import tqdm as tqdm_async
from .base import BaseVectorStorage
from .utils import logger, compute_mdhash_id


@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    """Хранилище векторов на базе Qdrant"""

    cosine_better_than_threshold: float = 0.7

    def __post_init__(self):
        """Инициализация подключения к Qdrant"""
        # Получаем параметры подключения из конфига
        qdrant_config = self.global_config.get("qdrant_config", {})
        self.host = qdrant_config.get("host", "localhost")
        self.port = qdrant_config.get("port", 6333)
        self.grpc_port = qdrant_config.get("grpc_port", 6334)
        self.api_key = qdrant_config.get("api_key", None)

        # Создаем клиент Qdrant
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            prefer_grpc=True,
            grpc_port=self.grpc_port,
        )

        # Имя коллекции
        self.collection_name = f"{self.global_config['working_dir'].replace('/', '_')}_{self.namespace}"

        # Размер батча для операций
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # Создаем коллекцию если её нет
        self._ensure_collection()

        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def _ensure_collection(self):
        """Создаем коллекцию если её нет"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except:
            # Создаем коллекцию
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_func.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection {self.collection_name}")

    async def upsert(self, data: dict[str, dict]):
        """Добавление или обновление векторов"""
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        # Подготавливаем контент для эмбеддингов
        contents = [v["content"] for v in data.values()]
        ids = list(data.keys())

        # Разбиваем на батчи
        batches = [
            contents[i: i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        # Генерируем эмбеддинги
        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        # Подготавливаем точки для Qdrant
        points = []
        for i, (id_, doc_data) in enumerate(data.items()):
            # Формируем payload с метаданными
            payload = {
                "content": doc_data["content"],
                "__id__": id_,
            }
            # Добавляем дополнительные мета-поля
            for field in self.meta_fields:
                if field in doc_data:
                    payload[field] = doc_data[field]

            point = PointStruct(
                id=id_,
                vector=embeddings[i].tolist(),
                payload=payload,
            )
            points.append(point)

        # Загружаем в Qdrant батчами
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Successfully inserted {len(points)} vectors")
        return ids

    async def query(self, query: str, top_k=5):
        """Поиск похожих векторов"""
        # Получаем эмбеддинг для запроса
        embedding = await self.embedding_func([query])
        embedding = embedding[0]

        # Выполняем поиск
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
            score_threshold=self.cosine_better_than_threshold,
        )

        # Формируем результаты
        results = []
        for hit in search_result:
            result = {
                "id": hit.payload.get("__id__", str(hit.id)),
                "distance": hit.score,
                "content": hit.payload.get("content", ""),
            }
            # Добавляем мета-поля
            for field in self.meta_fields:
                if field in hit.payload:
                    result[field] = hit.payload[field]
            results.append(result)

        return results

    async def delete_entity(self, entity_name: str):
        """Удаление сущности"""
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[entity_id]),
            )
            logger.info(f"Entity {entity_name} has been deleted.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        """Удаление всех связей сущности"""
        try:
            # Ищем все связи где участвует эта сущность
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="src_id",
                            match=models.MatchValue(value=entity_name),
                        ),
                        models.FieldCondition(
                            key="tgt_id",
                            match=models.MatchValue(value=entity_name),
                        ),
                    ]
                ),
                limit=10000,
            )

            ids_to_delete = [point.id for point in scroll_result[0]]

            if ids_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                )
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        """Вызывается после завершения индексации"""
        # Qdrant автоматически индексирует, поэтому здесь ничего не делаем
        pass