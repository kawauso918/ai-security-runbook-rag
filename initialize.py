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
from utils import chunk_by_headings, pdf_to_sections
from retriever import create_hybrid_retriever
from error_handler import DataFolderEmptyError, PDFReadError, display_error_summary


def load_documents(data_folder: str) -> Tuple[List[Dict], List[tuple[str, Exception]]]:
    """データフォルダからPDF/Markdownを読み込み
    
    Returns:
        (documents, errors) のタプル
        documents: 読み込み成功したドキュメントのリスト
        errors: (ファイルパス, エラー) のリスト
    """
    documents = []
    errors = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        raise DataFolderEmptyError(f"データフォルダ `{data_folder}` が存在しません。")
    
    # Markdownファイルを読み込み
    for md_file in data_path.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # 空ファイルはスキップ
                    documents.append({
                        'file_path': str(md_file),
                        'content': content,
                        'updated_at': datetime.fromtimestamp(md_file.stat().st_mtime).isoformat()
                    })
        except UnicodeDecodeError as e:
            errors.append((str(md_file), Exception(f"エンコーディングエラー: UTF-8で読み込めませんでした")))
        except Exception as e:
            errors.append((str(md_file), e))
    
    # テキストファイルも読み込み
    for txt_file in data_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # 空ファイルはスキップ
                    documents.append({
                        'file_path': str(txt_file),
                        'content': content,
                        'updated_at': datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
                    })
        except UnicodeDecodeError as e:
            errors.append((str(txt_file), Exception(f"エンコーディングエラー: UTF-8で読み込めませんでした")))
        except Exception as e:
            errors.append((str(txt_file), e))
    
    # PDFファイルの読み込み（pypdf使用、見出し推定→セクション化）
    try:
        from pypdf import PdfReader
        for pdf_file in data_path.glob("*.pdf"):
            try:
                # PDFを見出し推定→セクション化
                pdf_sections = pdf_to_sections(str(pdf_file))

                # 各セクションをDocumentとして追加
                for section in pdf_sections:
                    documents.append(section)

            except ValueError as e:
                # テキスト抽出できないPDF（画像のみ）
                error_msg = str(e)
                errors.append((str(pdf_file), PDFReadError(error_msg)))
            except Exception as e:
                errors.append((str(pdf_file), PDFReadError(f"PDF読み込みエラー: {e}")))
    except ImportError:
        # pypdfがインストールされていない場合、PDFファイルがあれば警告
        pdf_files = list(data_path.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files:
                errors.append((str(pdf_file), Exception("pypdfがインストールされていません。pip install pypdf を実行してください。")))
    
    if not documents and not errors:
        raise DataFolderEmptyError(f"データフォルダ `{data_folder}` に有効なファイルが見つかりません。")
    
    return documents, errors


def process_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """ドキュメントをチャンキング
    
    PDF由来のDocumentは既にセクション化されているため、
    セクション内でさらにチャンキングする
    """
    all_chunks = []
    
    for doc in documents:
        # PDF由来のDocument（page_start/page_endがある）の場合
        if 'page_start' in doc and 'page_end' in doc:
            # セクション内でチャンキング
            section_text = doc['content']
            section_chunks = _chunk_text(section_text, chunk_size, overlap)
            
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = f"{doc['file_path']}_p{doc['page_start']}-{doc['page_end']}_{i}"
                chunk = {
                    'chunk_id': chunk_id,
                    'text': chunk_text.strip(),
                    'heading': doc['heading'],
                    'file': doc['file_path'],
                    'updated_at': doc['updated_at'],
                    'chunk_index': len(all_chunks),
                    'page_start': doc['page_start'],
                    'page_end': doc['page_end']
                }
                all_chunks.append(chunk)
        else:
            # Markdown/TXT由来のDocument（従来通り）
            chunks = chunk_by_headings(
                doc['content'],
                doc['file_path'],
                chunk_size=chunk_size,
                overlap=overlap
            )
            all_chunks.extend(chunks)
    
    return all_chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """テキストを指定サイズでチャンキング（utils.pyから移植）"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # オーバーラップ処理
        if end < len(text) and overlap > 0:
            next_start = end - overlap
            for i in range(next_start, end):
                if text[i] in ['。', '\n', '.', '!', '?']:
                    next_start = i + 1
                    break
        
        chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
        
        if start >= len(text):
            break
    
    return chunks


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
        metadata = {
            'file': chunk['file'],
            'heading': chunk['heading'],
            'updated_at': chunk['updated_at'],
            'text': chunk['text']
        }
        # PDF由来の場合はページ情報を追加
        if 'page_start' in chunk:
            metadata['page_start'] = chunk['page_start']
        if 'page_end' in chunk:
            metadata['page_end'] = chunk['page_end']
        chunks_metadata[chunk['chunk_id']] = metadata

    # ベクトルストア構築
    texts = [chunk['text'] for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        meta = {
            'chunk_id': chunk['chunk_id'],
            'file': chunk['file'],
            'heading': chunk['heading'],
            'updated_at': chunk['updated_at']
        }
        # PDF由来の場合はページ情報を追加
        if 'page_start' in chunk:
            meta['page_start'] = chunk['page_start']
        if 'page_end' in chunk:
            meta['page_end'] = chunk['page_end']
        metadatas.append(meta)

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
        {
            'vectorstore': VectorStore,
            'hybrid_retriever': EnsembleRetriever,
            'chunks_metadata': Dict,
            'index_count': int,
            'index_last_built': str,
            'errors': List[tuple[str, Exception]]  # エラー情報
        }
    """
    # ドキュメント読み込み
    documents, errors = load_documents(data_folder)

    if not documents:
        return {
            'vectorstore': None,
            'hybrid_retriever': None,
            'chunks_metadata': {},
            'index_count': 0,
            'index_last_built': None,
            'errors': errors
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
        'index_last_built': datetime.now().isoformat(),
        'errors': errors
    }

