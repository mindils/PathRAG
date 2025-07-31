import asyncio
import aiohttp
import json
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


async def check_llm_model():
    """Проверка доступности LLM модели"""
    print("Checking LLM model at localhost:8801...")
    try:
        async with aiohttp.ClientSession() as session:
            # Тестовый запрос
            data = {
                "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "messages": [{"role": "user", "content": "Say 'hello'"}],
                "max_tokens": 10,
            }

            async with session.post(
                    "http://localhost:8801/v1/chat/completions",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✓ LLM model is working!")
                    print(f"  Response: {result['choices'][0]['message']['content']}")
                    return True
                else:
                    print(f"✗ LLM model returned status {response.status}")
                    return False
    except Exception as e:
        print(f"✗ LLM model is not accessible: {str(e)}")
        return False


async def check_embedding_model():
    """Проверка доступности embedding модели"""
    print("\nChecking embedding model at localhost:8802...")
    try:
        async with aiohttp.ClientSession() as session:
            # Тестовый запрос
            data = {
                "model": "Qwen/Qwen3-Embedding-8B",
                "input": "test",
                "encoding_format": "float"
            }

            async with session.post(
                    "http://localhost:8802/v1/embeddings",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding_dim = len(result["data"][0]["embedding"])
                    print("✓ Embedding model is working!")
                    print(f"  Embedding dimension: {embedding_dim}")
                    return True
                else:
                    print(f"✗ Embedding model returned status {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Embedding model is not accessible: {str(e)}")
        return False


def check_qdrant():
    """Проверка доступности Qdrant"""
    print("\nChecking Qdrant at localhost:6333...")
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print("✓ Qdrant is working!")
        print(f"  Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"✗ Qdrant is not accessible: {str(e)}")
        return False


def check_dependencies():
    """Проверка установленных зависимостей"""
    print("\nChecking dependencies...")

    required_packages = [
        "PathRag",
        "qdrant-client",
        "aiohttp",
        "numpy",
        "tiktoken",
        "networkx",
    ]

    optional_packages = [
        ("PyPDF2", "for PDF support"),
    ]

    all_good = True

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_good = False

    print("\nOptional dependencies:")
    for package, purpose in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed ({purpose})")
        except ImportError:
            print(f"ℹ {package} is NOT installed ({purpose})")

    return all_good


async def main():
    print("PathRAG Setup Verification")
    print("=" * 50)

    # Проверка зависимостей
    deps_ok = check_dependencies()

    # Проверка сервисов
    llm_ok = await check_llm_model()
    embed_ok = await check_embedding_model()
    qdrant_ok = check_qdrant()

    print("\n" + "=" * 50)
    print("Summary:")

    if deps_ok and llm_ok and embed_ok and qdrant_ok:
        print("✓ All systems are operational!")
        print("\nYou can now:")
        print("1. Place documents in the 'input' folder")
        print("2. Run: python index_documents.py")
        print("3. Run: python query_documents.py")
    else:
        print("✗ Some components are not working properly.")
        print("\nPlease ensure:")
        if not deps_ok:
            print("- All required packages are installed: pip install -e PathRAG")
        if not llm_ok:
            print("- LLM model is running on localhost:8801")
        if not embed_ok:
            print("- Embedding model is running on localhost:8802")
        if not qdrant_ok:
            print("- Qdrant is running on localhost:6333")
            print("  Install: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")


if __name__ == "__main__":
    asyncio.run(main())