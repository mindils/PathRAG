import aiohttp
import json
import numpy as np
from typing import Union, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .llm import GPTKeywordExtractionFormat
from .utils import wrap_embedding_func_with_attrs, logger


# Функция для работы с локальной LLM моделью
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
)
async def qwen_local_complete(
        prompt,
        system_prompt=None,
        history_messages=[],
        base_url="http://localhost:8801",
        **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Функция для работы с локальной Qwen моделью через API"""

    # Убираем специфичные параметры
    kwargs.pop("hashing_kv", None)
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    stream = kwargs.get("stream", False)

    # Формируем сообщения
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.info(f"!!!LLM message!!!: {messages}")


    # Подготавливаем данные для запроса
    data = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": messages,
        "stream": stream,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 2048),
    }

    logger.debug(f"Sending request to local LLM: {base_url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
                f"{base_url}/v1/chat/completions",
                json=data,
                headers={"Content-Type": "application/json"},
        ) as response:
            if stream:
                async def stream_generator():
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

                return stream_generator()
            else:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]

                # Обработка для извлечения ключевых слов
                if keyword_extraction:
                    try:
                        # Пытаемся найти JSON в ответе
                        import re
                        print("Hello world")
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            data = json.loads(json_str)
                            return json.dumps({
                                "high_level_keywords": data.get("high_level_keywords", []),
                                "low_level_keywords": data.get("low_level_keywords", [])
                            }, ensure_ascii=False)
                    except:
                        logger.warning("Failed to parse keyword extraction response")
                        return json.dumps({
                            "high_level_keywords": [],
                            "low_level_keywords": []
                        }, ensure_ascii=False)

                return content


# Функция для работы с локальной embedding моделью
@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
)
async def qwen_local_embedding(
        texts: list[str],
        base_url="http://localhost:8802",
        **kwargs,
) -> np.ndarray:
    """Функция для получения embeddings от локальной Qwen модели"""

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []

    async with aiohttp.ClientSession() as session:
        for text in texts:
            data = {
                "model": "Qwen/Qwen3-Embedding-8B",
                "input": text,
                "encoding_format": "float"
            }

            async with session.post(
                    f"{base_url}/v1/embeddings",
                    json=data,
                    headers={"Content-Type": "application/json"},
            ) as response:
                result = await response.json()
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)

    return np.array(embeddings)