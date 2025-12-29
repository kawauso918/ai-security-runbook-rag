"""ハイブリッド検索処理（EnsembleRetriever使用）"""

from typing import List, Dict, Any
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document

from utils import tokenize_japanese


def create_hybrid_retriever(
    vectorstore,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    bm25_weight: float = 0.6,
    vector_weight: float = 0.4,
    k: int = 4
) -> EnsembleRetriever:
    """ハイブリッドRetrieverを作成

    Args:
        vectorstore: ベクトルストア（Chroma等）
        texts: テキストのリスト
        metadatas: メタデータのリスト
        bm25_weight: BM25の重み（0.0-1.0）
        vector_weight: ベクトル検索の重み（0.0-1.0）
        k: 返す結果数

    Returns:
        EnsembleRetriever: ハイブリッド検索用のRetriever
    """
    # BM25Retrieverを作成
    # LangChainのBM25Retrieverにカスタムトークナイザーを設定
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]

    # BM25Retrieverを作成（カスタムトークナイザー使用）
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=k,
        preprocess_func=tokenize_japanese  # Sudachiトークナイザーを使用
    )

    # VectorRetrieverを作成
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # EnsembleRetrieverで統合
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
        k=k
    )

    return ensemble_retriever


def update_retriever_weights(
    ensemble_retriever: EnsembleRetriever,
    bm25_weight: float,
    vector_weight: float
) -> None:
    """EnsembleRetrieverの重みを更新

    Args:
        ensemble_retriever: 更新対象のEnsembleRetriever
        bm25_weight: BM25の重み
        vector_weight: ベクトル検索の重み
    """
    ensemble_retriever.weights = [bm25_weight, vector_weight]


def update_retriever_k(
    ensemble_retriever: EnsembleRetriever,
    k: int
) -> None:
    """EnsembleRetrieverのk値を更新

    Args:
        ensemble_retriever: 更新対象のEnsembleRetriever
        k: 返す結果数
    """
    ensemble_retriever.k = k
    # 各Retrieverのkも更新
    for retriever in ensemble_retriever.retrievers:
        if hasattr(retriever, 'k'):
            retriever.k = k
        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs['k'] = k


def search_with_scores(
    ensemble_retriever: EnsembleRetriever,
    query: str,
    k: int = 4
) -> List[Dict[str, Any]]:
    """スコア付きで検索を実行

    Args:
        ensemble_retriever: EnsembleRetriever
        query: 検索クエリ
        k: 返す結果数

    Returns:
        検索結果のリスト。各要素は以下の構造：
        {
            'chunk_id': str,
            'text': str,
            'score': float,  # 0.0-1.0の範囲で正規化されたスコア
            'file': str,
            'heading': str,
            'updated_at': str
        }
    """
    # k値を更新
    update_retriever_k(ensemble_retriever, k)

    # 検索実行（EnsembleRetrieverはget_relevant_documentsを使用）
    docs = ensemble_retriever.get_relevant_documents(query)

    # 結果を整形（スコアを追加）
    results = []
    for i, doc in enumerate(docs[:k]):
        # EnsembleRetrieverはスコアを直接返さないため、
        # ランキング順位からスコアを計算（1位=1.0, 2位=0.9, ...）
        score = 1.0 - (i * 0.1)  # 簡易的なスコア計算
        if score < 0.0:
            score = 0.0

        result = {
            'chunk_id': doc.metadata.get('chunk_id', ''),
            'text': doc.page_content,
            'score': score,
            'bm25_score': score,  # 互換性のため
            'vector_score': score,  # 互換性のため
            'file': doc.metadata.get('file', ''),
            'heading': doc.metadata.get('heading', ''),
            'updated_at': doc.metadata.get('updated_at', '')
        }
        results.append(result)

    return results


def hybrid_search(
    query: str,
    k: int,
    bm25_weight: float,
    vector_weight: float,
    vectorstore,
    bm25_index=None,  # 互換性のため残す（未使用）
    chunks_metadata=None  # 互換性のため残す（未使用）
) -> List[Dict[str, Any]]:
    """ハイブリッド検索実行（後方互換性のため残す）

    Note: この関数は後方互換性のために残していますが、
    新しいコードではcreate_hybrid_retriever + search_with_scoresを使用してください。

    Args:
        query: 検索クエリ
        k: 返す結果数
        bm25_weight: BM25の重み
        vector_weight: ベクトル検索の重み
        vectorstore: ベクトルストア
        bm25_index: （未使用、互換性のため）
        chunks_metadata: （未使用、互換性のため）

    Returns:
        検索結果のリスト
    """
    # この関数は既存のコードとの互換性のために残す
    # 実際には、main.pyでEnsembleRetrieverを直接使用することを推奨
    # 暫定的にベクトル検索のみを返す
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = vector_retriever.get_relevant_documents(query)

    results = []
    for i, doc in enumerate(docs[:k]):
        score = 1.0 - (i * 0.1)
        if score < 0.0:
            score = 0.0

        result = {
            'chunk_id': doc.metadata.get('chunk_id', ''),
            'text': doc.page_content,
            'score': score,
            'bm25_score': score,
            'vector_score': score,
            'file': doc.metadata.get('file', ''),
            'heading': doc.metadata.get('heading', ''),
            'updated_at': doc.metadata.get('updated_at', '')
        }
        results.append(result)

    return results
