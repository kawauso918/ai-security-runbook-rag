"""初期化処理（インデックス構築）"""

import os
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from constants import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL, CHROMA_DB_PATH,
    DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT, DEFAULT_K
)
from utils import chunk_by_headings, pdf_to_sections, pdf_to_sections_with_ocr
from retriever import create_hybrid_retriever
from error_handler import DataFolderEmptyError, PDFReadError, display_error_summary


def load_documents(
    data_folder: str,
    ocr_enabled: bool = False,
    ocr_method: str = "tesseract",
    ocr_language: str = "jpn"
) -> Tuple[List[Dict], List[tuple[str, Exception]]]:
    """データフォルダからPDF/Markdownを読み込み

    Args:
        data_folder: データフォルダパス
        ocr_enabled: OCR処理を有効化するか
        ocr_method: OCR手法（tesseract または azure）
        ocr_language: OCR言語（jpn, eng, jpn+eng等）

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
    
    # PDFファイルの読み込み（OCR対応、見出し推定→セクション化）
    try:
        from pypdf import PdfReader
        for pdf_file in data_path.glob("*.pdf"):
            try:
                # PDFを見出し推定→セクション化（OCR対応）
                pdf_sections = pdf_to_sections_with_ocr(
                    pdf_path=str(pdf_file),
                    ocr_enabled=ocr_enabled,
                    ocr_method=ocr_method,
                    ocr_language=ocr_language,
                    progress_callback=None  # 後で追加
                )

                # 各セクションをDocumentとして追加
                for section in pdf_sections:
                    documents.append(section)

            except ValueError as e:
                # テキスト抽出できないPDF（画像のみ）かつOCRも失敗
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
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    adaptive: bool = True
) -> List[Dict]:
    """ドキュメントをチャンキング（適応的チャンキング対応）

    PDF由来のDocumentは既にセクション化されているため、
    セクション内でさらにチャンキングする

    Args:
        documents: ドキュメントのリスト
        chunk_size: 基本チャンクサイズ（フォールバック用）
        overlap: オーバーラップサイズ
        adaptive: 適応的チャンキングを有効化（デフォルト: True）

    Returns:
        チャンクのリスト
    """
    from constants import CHUNK_SIZE_PDF, CHUNK_SIZE_MARKDOWN, CHUNK_SIZE_TEXT

    all_chunks = []

    for doc in documents:
        # ファイル拡張子からドキュメントタイプを判定
        file_path = doc['file_path']
        if file_path.endswith('.pdf'):
            doc_type = 'pdf'
            base_chunk_size = CHUNK_SIZE_PDF if adaptive else chunk_size
        elif file_path.endswith('.md'):
            doc_type = 'markdown'
            base_chunk_size = CHUNK_SIZE_MARKDOWN if adaptive else chunk_size
        else:
            doc_type = 'text'
            base_chunk_size = CHUNK_SIZE_TEXT if adaptive else chunk_size

        # PDF由来のDocument（page_start/page_endがある）の場合
        if 'page_start' in doc and 'page_end' in doc:
            # セクション内でチャンキング（適応的チャンキングパラメータを渡す）
            section_text = doc['content']
            section_chunks = _chunk_text(
                section_text,
                base_chunk_size,
                overlap,
                adaptive=adaptive,
                doc_type=doc_type
            )

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
            # Markdown/TXT由来のDocument（適応的チャンキングパラメータを渡す）
            chunks = chunk_by_headings(
                doc['content'],
                doc['file_path'],
                chunk_size=base_chunk_size,
                overlap=overlap,
                adaptive=adaptive,
                doc_type=doc_type
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


def _ensure_chroma_db_path(db_path: str) -> None:
    """ChromaDBの保存先を用意（削除せずに再利用）"""
    path = Path(db_path)
    parent = path.parent

    if not parent.exists():
        raise PermissionError(f"ChromaDBの保存先の親フォルダが存在しません: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(f"ChromaDBの保存先に書き込み権限がありません: {parent}")

    if path.exists():
        if not path.is_dir():
            raise PermissionError(f"ChromaDBの保存先がディレクトリではありません: {path}")
        if not os.access(path, os.W_OK | os.X_OK):
            raise PermissionError(f"ChromaDBの保存先に書き込み権限がありません: {path}")
        return

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise PermissionError(f"ChromaDBの保存先を作成できません: {path} ({e})")


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

    # ChromaDBの保存先を用意（削除せず再利用）
    _ensure_chroma_db_path(CHROMA_DB_PATH)

    # ChromaDB 0.4.0+ 用: PersistentClientを使用
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 既存のコレクションを削除（インデックス再構築時に古いデータを削除）
    try:
        chroma_client.delete_collection("security_runbooks")
    except Exception:
        pass

    # ChromaDBに保存（永続化）
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        client=chroma_client,
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
    k: int = DEFAULT_K,
    ocr_enabled: bool = False,
    ocr_method: str = "tesseract",
    ocr_language: str = "jpn"
) -> Dict:
    """システム全体の初期化

    Args:
        data_folder: データフォルダパス
        chunk_size: チャンクサイズ
        overlap: チャンクのオーバーラップサイズ
        bm25_weight: BM25の重み
        vector_weight: ベクトル検索の重み
        k: デフォルトの検索結果数
        ocr_enabled: OCR処理を有効化するか
        ocr_method: OCR手法（tesseract または azure）
        ocr_language: OCR言語（jpn, eng, jpn+eng等）

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
    # ドキュメント読み込み（OCR設定を渡す）
    documents, errors = load_documents(
        data_folder,
        ocr_enabled=ocr_enabled,
        ocr_method=ocr_method,
        ocr_language=ocr_language
    )

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
