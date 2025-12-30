"""Re-ranking前後の検索結果を比較する簡易評価スクリプト

使用方法:
    python eval/compare_reranking.py

環境変数:
    OPENAI_API_KEY: OpenAI APIキー（必須）
    COHERE_API_KEY: Cohere APIキー（Re-ranking有効時のみ必須）
"""

import json
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from initialize import initialize_system
from retriever import search_with_scores, rerank_search_results
from constants import (
    DEFAULT_DATA_FOLDER, DEFAULT_K, DEFAULT_BM25_WEIGHT,
    DEFAULT_VECTOR_WEIGHT, RERANK_TOP_K_BEFORE
)

# 環境変数読み込み
load_dotenv()


def run_comparison():
    """Re-ranking前後の検索結果を比較"""

    print("=" * 80)
    print("Re-ranking 前後比較評価")
    print("=" * 80)

    # 評価データセット読み込み
    eval_file = Path(__file__).parent / "eval_dataset.json"
    with open(eval_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Re-ranking効果測定用のケースを抽出
    rerank_cases = [tc for tc in test_cases if tc.get('category') == 'Re-ranking効果測定']
    if not rerank_cases:
        print("⚠️ Re-ranking効果測定用のテストケースが見つかりません")
        print("すべてのテストケースで評価します...")
        rerank_cases = test_cases[:5]  # 最初の5件

    print(f"\nテストケース数: {len(rerank_cases)}")

    # システム初期化
    print("\n初期化中...")
    try:
        init_result = initialize_system(
            data_folder=DEFAULT_DATA_FOLDER,
            bm25_weight=DEFAULT_BM25_WEIGHT,
            vector_weight=DEFAULT_VECTOR_WEIGHT,
            k=DEFAULT_K
        )

        if init_result['index_count'] == 0:
            print(f"❌ エラー: {DEFAULT_DATA_FOLDER} にインデックス可能なファイルがありません")
            return

        print(f"✅ インデックス構築完了: {init_result['index_count']} チャンク")

        hybrid_retriever = init_result['hybrid_retriever']

    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return

    # 比較結果を格納
    results = []

    for idx, test_case in enumerate(rerank_cases, 1):
        query = test_case['question']
        print(f"\n[{idx}/{len(rerank_cases)}] {query}")
        print("-" * 80)

        # Re-ranking無効で検索
        print("\n【Re-ranking無効】")
        try:
            results_without_rerank = search_with_scores(
                ensemble_retriever=hybrid_retriever,
                query=query,
                k=DEFAULT_K
            )

            print(f"検索結果: {len(results_without_rerank)}件")
            for i, result in enumerate(results_without_rerank[:3], 1):
                print(f"  {i}. スコア: {result['score']:.3f} | {result['heading'][:50]}")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            results_without_rerank = []

        # Re-ranking有効で検索（LLMベース）
        print("\n【Re-ranking有効（LLM）】")
        try:
            # より多くの結果を取得
            results_before_rerank = search_with_scores(
                ensemble_retriever=hybrid_retriever,
                query=query,
                k=RERANK_TOP_K_BEFORE
            )

            # Re-ranking適用（LLMベース - OpenAI APIを使用）
            results_with_rerank = rerank_search_results(
                query=query,
                search_results=results_before_rerank,
                k=DEFAULT_K,
                method="llm"
            )

            print(f"検索結果: {len(results_with_rerank)}件")
            for i, result in enumerate(results_with_rerank[:3], 1):
                rerank_score = result.get('rerank_score', 0.0)
                print(f"  {i}. Re-rankスコア: {rerank_score:.3f} | 元スコア: {result['score']:.3f} | {result['heading'][:50]}")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            if "COHERE_API_KEY" in str(e):
                print("  💡 ヒント: .envファイルにCOHERE_API_KEYを設定してください")
            results_with_rerank = []

        # 結果を記録
        results.append({
            'question': query,
            'without_rerank': [
                {
                    'heading': r['heading'],
                    'score': r['score'],
                    'file': r['file']
                }
                for r in results_without_rerank[:3]
            ],
            'with_rerank': [
                {
                    'heading': r['heading'],
                    'score': r['score'],
                    'rerank_score': r.get('rerank_score', 0.0),
                    'file': r['file']
                }
                for r in results_with_rerank[:3]
            ]
        })

    # 結果をファイルに保存
    output_file = Path(__file__).parent / f"rerank_comparison_{Path(DEFAULT_DATA_FOLDER).name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ 比較結果を保存しました: {output_file}")
    print("=" * 80)

    # サマリー表示
    print("\n【サマリー】")
    print(f"- テストケース数: {len(rerank_cases)}")
    print(f"- Re-ranking無効の平均スコア: {_calc_avg_score([r['without_rerank'] for r in results]):.3f}")
    print(f"- Re-ranking有効の平均Re-rankスコア: {_calc_avg_rerank_score([r['with_rerank'] for r in results]):.3f}")
    print("\n💡 次のステップ:")
    print("  1. 結果ファイルを確認して、Re-ranking効果を分析")
    print("  2. より詳細な評価を行う場合は、LLM as a Judge評価を実行")


def _calc_avg_score(results_list):
    """平均スコアを計算"""
    scores = []
    for results in results_list:
        for result in results:
            scores.append(result.get('score', 0.0))
    return sum(scores) / len(scores) if scores else 0.0


def _calc_avg_rerank_score(results_list):
    """平均Re-rankスコアを計算"""
    scores = []
    for results in results_list:
        for result in results:
            scores.append(result.get('rerank_score', 0.0))
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    run_comparison()
