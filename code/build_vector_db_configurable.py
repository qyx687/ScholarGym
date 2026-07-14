#!/usr/bin/env python3
"""Build a Qdrant collection with the exact embedding backend used at run time."""

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

import config
from graph_methods import normalize_arxiv_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_db", required=True)
    parser.add_argument("--qdrant_url", default=config.QDRANT_URL)
    parser.add_argument("--qdrant_collection", default="paper_knowledge_base")
    parser.add_argument("--embedding_base_url", default=config.OLLAMA_URL)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    provider = OllamaEmbeddings(
        model=getattr(config, "OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        base_url=args.embedding_base_url,
    )
    probe = provider.embed_query("dimension probe")
    client = QdrantClient(url=args.qdrant_url)
    exists = client.collection_exists(args.qdrant_collection)
    if exists and args.recreate:
        client.delete_collection(args.qdrant_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=args.qdrant_collection,
            vectors_config=VectorParams(size=len(probe), distance=Distance.COSINE),
        )
    store = QdrantVectorStore(client=client, collection_name=args.qdrant_collection, embedding=provider)
    value = json.loads(Path(args.paper_db).read_text(encoding="utf-8"))
    items = value.items() if isinstance(value, dict) else enumerate(value)
    texts, metadatas, ids = [], [], []
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    for key, paper in items:
        if not isinstance(paper, dict):
            continue
        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id") or key)
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        # Match ScholarGym's baseline build_vector_db.py serialization exactly.
        text = f"title: {title}\n abstract: {abstract}" if title or abstract else ""
        if not arxiv_id or not text:
            continue
        texts.append(text)
        metadatas.append(
            {
                "arxiv_id": arxiv_id,
                "id": arxiv_id,
                "title": paper.get("title") or "",
                "abstract": paper.get("abstract") or "",
                "date": paper.get("date") or "",
            }
        )
        ids.append(str(uuid.uuid5(namespace, arxiv_id)))
        if len(texts) >= args.batch_size:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
            texts, metadatas, ids = [], [], []
    if texts:
        store.add_texts(texts, metadatas=metadatas, ids=ids)
    print(f"Qdrant collection ready: {args.qdrant_collection}")


if __name__ == "__main__":
    main()
