"""初期化処理（インデックス構築）"""

import os
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from constants import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL, CHROMA_DB_PATH,
    DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT, DEFAULT_K
)
from utils import chunk_by_headings
from retriever import create_hybrid_retriever


def load_documents(data_folder: str) -> List[Dict]:
    """データフォルダからPDF/Markdownを読み込み"""
    documents = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        return documents
    
    # Markdownファイルを読み込み
    for md_file in data_path.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'file_path': str(md_file),
                    'content': content,
                    'updated_at': datetime.fromtimestamp(md_file.stat().st_mtime).isoformat()
                })
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    
    # テキストファイルも読み込み
    for txt_file in data_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'file_path': str(txt_file),
                    'content': content,
                    'updated_at': datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
                })
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    # PDFファイルの読み込み（pypdf使用）
    try:
        from pypdf import PdfReader
        for pdf_file in data_path.glob("*.pdf"):
            try:
                reader = PdfReader(pdf_file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                
                documents.append({
                    'file_path': str(pdf_file),
                    'content': content,
                    'updated_at': datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                print(f"Error reading {pdf_file}: {e}")
    except ImportError:
        print("pypdf not available, skipping PDF files")
    
    return documents


def process_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """ドキュメントをチャンキング"""
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_by_headings(
            doc['content'],
            doc['file_path'],
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(chunks)
    
    return all_chunks


def build_indexes(
    chunks: List[Dict],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    k: int = DEFAULT_K
) -> Tuple:
    """ベクトルDBとハイブリッドRetrieverの構築

    Args:
        chunks: チャンクのリスト
        embedding_model: 埋め込みモデル名
        bm25_weight: BM25の重み
        vector_weight: ベクトル検索の重み
        k: デフォルトの検索結果数

    Returns:
        (vectorstore, hybrid_retriever, chunks_metadata)
    """
    # メタデータ辞書を作成
    chunks_metadata = {}
    for chunk in chunks:
        chunks_metadata[chunk['chunk_id']] = {
            'file': chunk['file'],
            'heading': chunk['heading'],
            'updated_at': chunk['updated_at'],
            'text': chunk['text']
        }

    # ベクトルストア構築
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            'chunk_id': chunk['chunk_id'],
            'file': chunk['file'],
            'heading': chunk['heading'],
            'updated_at': chunk['updated_at']
        }
        for chunk in chunks
    ]

    embeddings = OpenAIEmbeddings(model=embedding_model)

    # ChromaDBに保存（永続化）
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="security_runbooks"
    )

    # ハイブリッドRetrieverを作成
    hybrid_retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        texts=texts,
        metadatas=metadatas,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        k=k
    )

    return vectorstore, hybrid_retriever, chunks_metadata


def initialize_system(
    data_folder: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    k: int = DEFAULT_K
) -> Dict:
    """システム全体の初期化

    Args:
        data_folder: データフォルダパス
        chunk_size: チャンクサイズ
        overlap: チャンクのオーバーラップサイズ
        bm25_weight: BM25の重み
        vector_weight: ベクトル検索の重み
        k: デフォルトの検索結果数

    Returns:
        初期化結果の辞書
    """
    # ドキュメント読み込み
    documents = load_documents(data_folder)

    if not documents:
        return {
            'vectorstore': None,
            'hybrid_retriever': None,
            'chunks_metadata': {},
            'index_count': 0,
            'index_last_built': None
        }

    # チャンキング
    chunks = process_documents(documents, chunk_size=chunk_size, overlap=overlap)

    # インデックス構築
    vectorstore, hybrid_retriever, chunks_metadata = build_indexes(
        chunks,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        k=k
    )

    return {
        'vectorstore': vectorstore,
        'hybrid_retriever': hybrid_retriever,
        'chunks_metadata': chunks_metadata,
        'index_count': len(chunks),
        'index_last_built': datetime.now().isoformat()
    }

