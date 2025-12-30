"""ハイブリッド検索処理（EnsembleRetriever使用）"""

from typing import List, Dict, Any
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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
    # EnsembleRetrieverのweightsを更新（Pydanticモデルの場合は直接設定可能）
    if hasattr(ensemble_retriever, 'weights'):
        # weightsがリストの場合
        if isinstance(ensemble_retriever.weights, list):
            ensemble_retriever.weights = [bm25_weight, vector_weight]
        # weightsが属性として設定可能な場合
        else:
            try:
                object.__setattr__(ensemble_retriever, 'weights', [bm25_weight, vector_weight])
            except:
                # 更新できない場合はスキップ（初期化時に設定されているため）
                pass


def update_retriever_k(
    ensemble_retriever: EnsembleRetriever,
    k: int
) -> None:
    """EnsembleRetrieverのk値を更新

    Args:
        ensemble_retriever: 更新対象のEnsembleRetriever
        k: 返す結果数
    """
    # EnsembleRetriever自体にはkフィールドがないため、各Retrieverのkを更新
    for retriever in ensemble_retriever.retrievers:
        # BM25Retrieverの場合
        if hasattr(retriever, 'k'):
            try:
                retriever.k = k
            except:
                # 更新できない場合はスキップ
                pass
        # VectorRetrieverの場合
        if hasattr(retriever, 'search_kwargs'):
            if isinstance(retriever.search_kwargs, dict):
                retriever.search_kwargs['k'] = k
            else:
                try:
                    # search_kwargsが属性の場合
                    object.__setattr__(retriever, 'search_kwargs', {'k': k})
                except:
                    pass


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
    # k値を更新（各Retrieverの設定を更新）
    # 注意: EnsembleRetriever自体にはkフィールドがないため、各Retrieverのkを更新
    for retriever in ensemble_retriever.retrievers:
        # BM25Retrieverの場合
        if hasattr(retriever, 'k'):
            try:
                retriever.k = k
            except:
                pass
        # VectorRetrieverの場合
        if hasattr(retriever, 'search_kwargs'):
            if isinstance(retriever.search_kwargs, dict):
                retriever.search_kwargs['k'] = k

    # 検索実行（EnsembleRetrieverはinvokeを使用）
    docs = ensemble_retriever.invoke(query)

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
        # PDF由来の場合はページ情報を追加
        if 'page_start' in doc.metadata:
            result['page_start'] = doc.metadata['page_start']
        if 'page_end' in doc.metadata:
            result['page_end'] = doc.metadata['page_end']
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

    docs = vector_retriever.invoke(query)

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
        # PDF由来の場合はページ情報を追加
        if 'page_start' in doc.metadata:
            result['page_start'] = doc.metadata['page_start']
        if 'page_end' in doc.metadata:
            result['page_end'] = doc.metadata['page_end']
        results.append(result)

    return results


def fallback_keyword_search(
    query: str,
    chunks_metadata: Dict[str, Dict[str, Any]],
    k: int = 4
) -> List[Dict[str, Any]]:
    """インデックス結果が空の場合のキーワード簡易検索"""
    if not chunks_metadata:
        return []

    tokens = [t.strip() for t in tokenize_japanese(query) if t.strip()]
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return []

    scored = []
    for chunk_id, meta in chunks_metadata.items():
        text = meta.get('text', '')
        if not text:
            continue
        match_count = 0
        for token in tokens:
            if token in text:
                match_count += 1
        if match_count <= 0:
            continue
        scored.append((chunk_id, meta, match_count))

    if not scored:
        return []

    max_score = max(score for _, _, score in scored)
    results = []
    for chunk_id, meta, score in sorted(scored, key=lambda x: x[2], reverse=True)[:k]:
        normalized = score / max_score if max_score else 0.0
        result = {
            'chunk_id': chunk_id,
            'text': meta.get('text', ''),
            'score': min(1.0, max(0.1, normalized)),
            'bm25_score': min(1.0, max(0.1, normalized)),
            'vector_score': 0.0,
            'file': meta.get('file', ''),
            'heading': meta.get('heading', ''),
            'updated_at': meta.get('updated_at', '')
        }
        if 'page_start' in meta:
            result['page_start'] = meta['page_start']
        if 'page_end' in meta:
            result['page_end'] = meta['page_end']
        results.append(result)

    return results


# ==================== Re-ranking 関数 ====================

def rerank_search_results(
    query: str,
    search_results: List[Dict[str, Any]],
    k: int = 4,
    method: str = "cohere",
    model: str = None
) -> List[Dict[str, Any]]:
    """検索結果を再ランキング

    Args:
        query: ユーザーの質問
        search_results: ハイブリッド検索の結果
        k: 返却する結果数
        method: Re-ranking手法（'cohere', 'llm', 'none'）
        model: LLMモデル名（method='llm'の場合）

    Returns:
        再ランキング後の検索結果（上位k件）
        各結果に 'rerank_score' フィールドが追加される
    """
    if method == "none" or not search_results:
        return search_results[:k]

    try:
        if method == "cohere":
            reranked_results = _rerank_with_cohere(query, search_results, k)
        elif method == "llm":
            reranked_results = _rerank_with_llm(query, search_results, k, model)
        else:
            # 不明なメソッドの場合は元の結果を返す
            return search_results[:k]

        return reranked_results
    except Exception as e:
        # Re-ranking失敗時は元の結果を返す（フォールバック）
        print(f"Re-ranking失敗: {e}")
        return search_results[:k]


def _rerank_with_cohere(
    query: str,
    search_results: List[Dict[str, Any]],
    k: int = 4,
    api_key: str = None
) -> List[Dict[str, Any]]:
    """Cohere Rerank APIで再ランキング

    Args:
        query: ユーザーの質問
        search_results: 検索結果
        k: 返却する結果数
        api_key: Cohere APIキー（環境変数から取得可能）

    Returns:
        再ランキング後の検索結果（rerank_scoreを含む）
    """
    import os

    try:
        import cohere
    except ImportError:
        raise ImportError(
            "cohereパッケージがインストールされていません。\n"
            "インストール方法: pip install cohere"
        )

    # 環境変数から取得
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "Cohere APIキーが設定されていません。\n"
            ".envファイルにCOHERE_API_KEYを追加してください。"
        )

    # Cohere Client
    co = cohere.Client(api_key)

    # 検索結果のテキストを抽出
    documents = [result['text'] for result in search_results]

    # Rerank API呼び出し
    rerank_response = co.rerank(
        query=query,
        documents=documents,
        top_n=min(k, len(documents)),
        model="rerank-multilingual-v3.0"  # 日本語対応モデル
    )

    # 結果を再構築
    reranked_results = []
    for item in rerank_response.results:
        result = search_results[item.index].copy()
        result['rerank_score'] = item.relevance_score
        reranked_results.append(result)

    return reranked_results


def _rerank_with_llm(
    query: str,
    search_results: List[Dict[str, Any]],
    k: int = 4,
    model: str = None
) -> List[Dict[str, Any]]:
    """LLMで再ランキング

    Args:
        query: ユーザーの質問
        search_results: 検索結果
        k: 返却する結果数
        model: LLMモデル名

    Returns:
        再ランキング後の検索結果（rerank_scoreを含む）
    """
    import json
    from openai import OpenAI
    from constants import RERANK_LLM_MODEL

    model = model or RERANK_LLM_MODEL

    client = OpenAI()

    # プロンプト構築
    prompt = f"""以下の検索結果について、質問との関連性を0-100点で評価してください。

【質問】
{query}

【検索結果】
"""
    for i, result in enumerate(search_results):
        text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
        prompt += f"\n[結果{i+1}]\n{text_preview}\n"

    prompt += """
【出力形式】
JSON形式で各結果のスコアを出力してください。
{
    "scores": [85, 70, 45, ...],
    "reasoning": "評価理由"
}
"""

    # LLM呼び出し
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        # スコアを取得
        result_data = json.loads(response.choices[0].message.content)
        scores = result_data.get("scores", [])

        # 結果にスコアを付与
        for i, result in enumerate(search_results):
            result['rerank_score'] = scores[i] / 100.0 if i < len(scores) else 0.0

        # スコア降順でソート
        reranked_results = sorted(
            search_results,
            key=lambda x: x.get('rerank_score', 0.0),
            reverse=True
        )

        return reranked_results[:k]

    except Exception as e:
        # LLM呼び出し失敗時は元の結果を返す
        print(f"LLMベースReranking失敗: {e}")
        return search_results[:k]
