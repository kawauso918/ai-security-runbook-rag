# Re-ranking機能 実装完了レポート

**実装日**: 2025-12-30
**対象機能**: Re-ranking（検索精度向上）
**実装方式**: Cohere Rerank API（優先）/ LLMベースReranking（代替）

---

## ✅ 実装完了項目

### 1. **constants.py にRe-ranking設定を追加**
- `RERANK_ENABLED`: デフォルトでRe-rankingを無効化（オプトイン）
- `RERANK_METHOD`: 'cohere', 'llm', 'none'
- `RERANK_LLM_MODEL`: LLMベースReranking用モデル
- `RERANK_TOP_K_BEFORE`: Re-ranking前に取得する結果数

### 2. **retriever.py にRe-ranking関数を実装**

#### `rerank_search_results()`
- 検索結果を再ランキング
- method='cohere', 'llm', 'none'に対応
- フォールバック機能（Re-ranking失敗時は元の結果を返す）

#### `_rerank_with_cohere()`
- Cohere Rerank APIで再ランキング
- `rerank-multilingual-v3.0`モデルを使用（日本語対応）
- relevance_scoreを`rerank_score`として各結果に付与

#### `_rerank_with_llm()`
- OpenAI LLMで再ランキング
- `gpt-4o-mini`を使用してコスト効率を確保
- JSONフォーマットで各結果のスコアを取得

### 3. **main.py のhandle_query()にRe-rankingを統合**

**変更点**:
```python
# Re-rankingが有効な場合は、より多くの結果を取得（k' > k）
k_before_rerank = session_state.get('rerank_top_k_before', 10) \
    if session_state.get('rerank_enabled', False) \
    else session_state['k']

search_results = search_with_scores(
    ensemble_retriever=session_state['hybrid_retriever'],
    query=user_query,
    k=k_before_rerank
)

# Re-ranking適用
if session_state.get('rerank_enabled', False) and search_results:
    search_results = rerank_search_results(...)
```

### 4. **UIにRe-ranking設定を追加（サイドバー）**

**追加UI**:
- Re-ranking有効化チェックボックス
- Re-ranking手法選択（cohere / llm / none）
- Re-ranking前の取得数（k'）設定
- Cohere APIキー設定のヒント表示

### 5. **logger.py にrerank_scoreを記録**

**ログに追加される情報**:
- 各検索結果の`rerank_score`（Re-ranking適用時のみ）
- Re-ranking設定情報（`rerank.enabled`, `rerank.method`）

**ログ形式**:
```json
{
  "rerank": {
    "enabled": true,
    "method": "cohere"
  },
  "sources": [
    {
      "chunk_id": "...",
      "score": 0.85,
      "rerank_score": 0.92,
      ...
    }
  ]
}
```

### 6. **requirements.txt にcohereパッケージを追加**
```
# Re-ranking
cohere>=5.0.0  # Optional: Cohere Rerank API for improved search accuracy
```

### 7. **README.md に短期拡張案のセクションを統合**
- 詳細な実装計画へのリンク（SHORT_TERM_EXTENSIONS.md）
- Re-ranking機能の概要と期待効果を記載

---

## 🚀 セットアップ手順

### Cohere Rerank API（推奨）

1. **Cohere APIキーの取得**
   - https://dashboard.cohere.com/ でアカウント作成
   - APIキーを取得（無料プラン: 月1,000リクエストまで）

2. **パッケージのインストール**
   ```bash
   pip install cohere
   ```

3. **環境変数の設定**
   `.env`ファイルに追加:
   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   ```

4. **動作確認**
   ```python
   import cohere
   import os

   co = cohere.Client(os.getenv("COHERE_API_KEY"))
   response = co.rerank(
       query="テスト",
       documents=["doc1", "doc2"],
       top_n=2,
       model="rerank-multilingual-v3.0"
   )
   print(response.results)
   ```

### LLMベースReranking（代替案）

- 追加のセットアップ不要（既存のOpenAI APIキーを使用）
- コスト: Cohere無料枠を超える場合に検討

---

## 📊 使用方法

### 1. Streamlitアプリの起動

```bash
streamlit run main.py
```

### 2. サイドバーでRe-ranking設定

- **Re-rankingを有効化**: チェックを入れる
- **Re-ranking手法**: `cohere`を選択（推奨）
- **Re-ranking前の取得数**: 10（デフォルト）

### 3. 質問を入力

通常通り質問を入力すると、自動的にRe-rankingが適用されます。

### 4. ログで確認

```bash
# 今日のログを確認
cat logs/query_$(date +%Y-%m-%d).jsonl | jq .
```

`rerank_score`フィールドが各検索結果に追加されていることを確認できます。

---

## 🔍 評価方法

### Re-ranking前後の精度比較

**評価指標**:
- LLM as a Judge評価（根拠性・正確性スコア）
- Re-ranking前後の平均スコア差
- ユーザーフィードバック（主観評価）

**評価手順**:
1. Re-ranking無効で10問のテストケースを実行
2. Re-ranking有効で同じ10問を実行
3. LLM as a Judge評価を実行
4. 平均スコアを比較

**成功基準**:
- Re-ranking有効時の平均スコアが+5点以上向上
- 無関係な結果が上位に来るケースが減少

---

## ⚠️ 注意事項

### 1. Cohere API制限
- 無料プラン: 月1,000リクエストまで
- 超過した場合はエラーが発生（フォールバック: 元の結果を返す）

### 2. LLMベースRerankingのコスト
- 1質問あたり約$0.001-0.002（gpt-4o-mini使用）
- 大量のクエリでは Cohere が推奨

### 3. 処理時間
- Re-rankingにより処理時間が+0.5-1秒増加する可能性あり
- 許容できない場合はRe-rankingを無効化

---

## 🐛 トラブルシューティング

### Cohere APIエラー

**エラー**: `ValueError: Cohere APIキーが設定されていません`

**対応**:
1. `.env`ファイルに`COHERE_API_KEY`を追加
2. Streamlitアプリを再起動

### Re-ranking失敗

**ログに表示**: `Re-ranking失敗（元の結果を使用）: ...`

**対応**:
- ネットワーク接続を確認
- APIキーの有効性を確認
- Cohere APIの利用上限を確認

### Re-ranking結果が変わらない

**対応**:
1. サイドバーで「Re-rankingを有効化」がチェックされているか確認
2. ログで`rerank.enabled: true`になっているか確認
3. 検索結果が少ない（k=2など）場合、Re-ranking効果が小さい可能性あり

---

## 📈 今後の改善案

### 1. Re-rankingモデルの選択肢拡大
- Cohere `rerank-english-v3.0`（英語専用、高速）
- ローカルモデル（sentence-transformers）

### 2. Re-rankingキャッシュ
- 同じクエリに対してRe-ranking結果をキャッシュ
- 処理時間とコストを削減

### 3. Re-ranking評価ダッシュボード
- Re-ranking前後のスコア分布を可視化
- A/Bテスト機能

---

## 📝 変更ファイル一覧

| ファイル | 変更内容 |
|---------|----------|
| `constants.py` | Re-ranking設定定数を追加 |
| `retriever.py` | Re-ranking関数3つを実装 |
| `main.py` | handle_query()にRe-ranking統合、UIにRe-ranking設定追加 |
| `logger.py` | rerank_scoreとReranking設定をログに記録 |
| `requirements.txt` | cohere>=5.0.0を追加 |
| `README.md` | 短期拡張案のセクションを統合 |

---

**実装完了**: ✅ すべてのタスクが完了しました
**次のステップ**: フェーズ2（OCR対応）の実装に進むか、Re-ranking機能の評価を実施してください

---

**作成者**: Claude Sonnet 4.5
**バージョン**: 1.0
**最終更新**: 2025-12-30
