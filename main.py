"""Streamlitメインアプリケーション"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from constants import (
    DEFAULT_DATA_FOLDER, DEFAULT_K, DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT,
    DEFAULT_LLM_MODEL, MAX_CONVERSATION_HISTORY, DEFAULT_JUDGE_MODEL
)
from initialize import initialize_system
from retriever import search_with_scores, update_retriever_weights, update_retriever_k
from guardrails import apply_guardrails
from logger import log_query
from components import (
    render_citation, render_danger_banner, render_security_notice,
    render_chat_message
)
from error_handler import (
    DataFolderEmptyError, PDFReadError, IndexNotBuiltError, APIError,
    handle_data_folder_empty, handle_pdf_read_error, handle_index_not_built,
    handle_api_error, display_error_summary
)
from judge import (
    load_eval_dataset, run_evaluation_suite, save_evaluation_results,
    format_evaluation_summary
)

# 環境変数読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="AIセキュリティ運用手順書アシスタント",
    page_icon="🔒",
    layout="wide"
)

# セッション状態の初期化
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'hybrid_retriever' not in st.session_state:
    st.session_state.hybrid_retriever = None

if 'chunks_metadata' not in st.session_state:
    st.session_state.chunks_metadata = {}

if 'data_folder' not in st.session_state:
    st.session_state.data_folder = DEFAULT_DATA_FOLDER

if 'k' not in st.session_state:
    st.session_state.k = DEFAULT_K

if 'bm25_weight' not in st.session_state:
    st.session_state.bm25_weight = DEFAULT_BM25_WEIGHT

if 'vector_weight' not in st.session_state:
    st.session_state.vector_weight = DEFAULT_VECTOR_WEIGHT

if 'index_last_built' not in st.session_state:
    st.session_state.index_last_built = None

if 'index_count' not in st.session_state:
    st.session_state.index_count = 0

if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if 'eval_running' not in st.session_state:
    st.session_state.eval_running = False

if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None


def handle_query(user_query: str, session_state: Dict) -> Dict:
    """質問処理のオーケストレーション"""
    start_time = time.time()

    # インデックス未構築チェック
    if session_state['hybrid_retriever'] is None:
        raise IndexNotBuiltError("インデックスが構築されていません。")

    # Retrieverの重みとkを更新
    update_retriever_weights(
        session_state['hybrid_retriever'],
        session_state['bm25_weight'],
        session_state['vector_weight']
    )
    update_retriever_k(
        session_state['hybrid_retriever'],
        session_state['k']
    )

    # 検索実行
    try:
        search_results = search_with_scores(
            ensemble_retriever=session_state['hybrid_retriever'],
            query=user_query,
            k=session_state['k']
        )
    except Exception as e:
        raise APIError(f"検索処理エラー: {e}")
    
    # ガードレール適用（検索後、LLM呼び出し前）
    guardrail_result = apply_guardrails(
        query=user_query,
        search_results=search_results,
        answer=None  # まだ回答は生成していない
    )
    
    # 根拠不足または曖昧質問の場合は、ここで終了
    if not guardrail_result['should_respond']:
        processing_time = time.time() - start_time
        
        return {
            'answer': guardrail_result['answer'],
            'citations': guardrail_result['citations'],
            'flags': guardrail_result['flags'],
            'warning_reason': guardrail_result['warning_reason'],
            'top_score': guardrail_result['top_score'],
            'processing_time': processing_time,
            'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'cost': 0.0
        }
    
    # LLMプロンプト作成
    context_text = "\n\n".join([
        f"【{i+1}】{result['text']}\n（出典: {result['file']} > {result['heading']}）"
        for i, result in enumerate(search_results[:session_state['k']])
    ])
    
    # 会話履歴を取得（最新5往復）
    conversation_history = session_state['messages'][-MAX_CONVERSATION_HISTORY * 2:]
    history_text = ""
    for msg in conversation_history:
        if msg['role'] == 'user':
            history_text += f"ユーザー: {msg['content']}\n"
        elif msg['role'] == 'assistant':
            history_text += f"アシスタント: {msg['content']}\n"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """あなたはSOC（セキュリティ運用センター）の運用手順書アシスタントです。
提供された手順書の内容に基づいて、正確で分かりやすい回答をしてください。

重要なルール:
- 手順書の内容のみを根拠として回答してください
- 推測や憶測は避け、手順書に記載されている情報のみを使用してください
- 回答の最後に、参考にした手順書のセクションを明記してください
- 分からない場合は「該当する手順が見つかりませんでした」と回答してください"""),
        ("human", """以下の手順書の内容を参考に、質問に回答してください。

【手順書の内容】
{context}

【会話履歴】
{history}

【質問】
{question}

【回答】""")
    ])
    
    # LLM初期化
    llm = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=0.0
    )
    
    # プロンプト実行（ストリーミング対応）
    prompt = prompt_template.format_messages(
        context=context_text,
        history=history_text,
        question=user_query
    )
    
    # API呼び出し（リトライ対応）
    max_retries = 3
    retry_count = 0
    response = None
    
    while retry_count < max_retries:
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            retry_count += 1
            should_retry, error_msg = handle_api_error(e, retry_count)
            if not should_retry:
                raise APIError(error_msg)
            if retry_count < max_retries:
                time.sleep(2 ** retry_count)  # 指数バックオフ
            else:
                raise APIError(f"API呼び出しが{max_retries}回失敗しました: {e}")
    
    answer = response.content.strip()
    
    # ガードレール適用（回答生成後、危険操作検知）
    guardrail_result = apply_guardrails(
        query=user_query,
        search_results=search_results,
        answer=answer  # 生成した回答をチェック
    )
    
    # トークン使用量とコスト計算
    from utils import calculate_cost
    
    # レスポンスからトークン使用量を取得
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        token_usage = response.response_metadata['token_usage']
    else:
        # フォールバック: 概算
        estimated_prompt_tokens = len(context_text.split()) + len(user_query.split()) * 2
        estimated_completion_tokens = len(answer.split()) * 1.3
        token_usage = {
            'prompt_tokens': int(estimated_prompt_tokens),
            'completion_tokens': int(estimated_completion_tokens),
            'total_tokens': int(estimated_prompt_tokens + estimated_completion_tokens)
        }
    
    cost = calculate_cost(
        token_usage.get('prompt_tokens', 0),
        token_usage.get('completion_tokens', 0),
        DEFAULT_LLM_MODEL
    )
    
    processing_time = time.time() - start_time
    
    return {
        'answer': guardrail_result['answer'],
        'citations': guardrail_result['citations'],
        'flags': guardrail_result['flags'],
        'warning_reason': guardrail_result['warning_reason'],
        'top_score': guardrail_result['top_score'],
        'processing_time': processing_time,
        'token_usage': token_usage,
        'cost': cost
    }


def render_sidebar():
    """サイドバー描画"""
    with st.sidebar:
        st.title("⚙️ 設定")
        
        # データフォルダパス
        data_folder = st.text_input(
            "データフォルダパス",
            value=st.session_state.data_folder,
            help="手順書ファイル（PDF/Markdown）を格納しているフォルダのパス"
        )
        st.session_state.data_folder = data_folder
        
        # k設定
        k = st.number_input(
            "検索結果数 (k)",
            min_value=1,
            max_value=20,
            value=st.session_state.k,
            help="検索結果として取得するチャンク数"
        )
        st.session_state.k = int(k)
        
        # 重み設定
        st.subheader("検索重み")
        bm25_weight = st.slider(
            "BM25重み",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.bm25_weight,
            step=0.1,
            help="BM25検索の重み（0.0-1.0）"
        )
        st.session_state.bm25_weight = bm25_weight
        st.session_state.vector_weight = 1.0 - bm25_weight
        
        st.write(f"ベクトル重み: {st.session_state.vector_weight:.1f}")
        
        st.divider()
        
        # インデックス再構築ボタン
        if st.button("🔄 インデックス再構築", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📂 データフォルダを確認中...")
                progress_bar.progress(10)
                
                status_text.text("📄 ドキュメントを読み込み中...")
                progress_bar.progress(30)
                
                result = initialize_system(
                    st.session_state.data_folder,
                    bm25_weight=st.session_state.bm25_weight,
                    vector_weight=st.session_state.vector_weight,
                    k=st.session_state.k
                )
                
                status_text.text("🔍 チャンキング中...")
                progress_bar.progress(50)
                
                status_text.text("💾 ベクトルDBを構築中...")
                progress_bar.progress(70)
                
                status_text.text("🔎 BM25インデックスを構築中...")
                progress_bar.progress(90)
                
                # セッション状態を更新
                st.session_state.vectorstore = result['vectorstore']
                st.session_state.hybrid_retriever = result['hybrid_retriever']
                st.session_state.chunks_metadata = result['chunks_metadata']
                st.session_state.index_count = result['index_count']
                st.session_state.index_last_built = result['index_last_built']
                
                progress_bar.progress(100)
                status_text.text("✅ 完了")
                
                st.success(f"インデックス構築完了: {result['index_count']}件のチャンクをインデックス化しました")
                
                # エラーがある場合は表示
                if result.get('errors'):
                    display_error_summary(result['errors'])
                    
            except DataFolderEmptyError as e:
                progress_bar.empty()
                status_text.empty()
                handle_data_folder_empty(st.session_state.data_folder)
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ インデックス構築中にエラーが発生しました: {e}")
                import traceback
                with st.expander("エラー詳細", expanded=False):
                    st.code(traceback.format_exc())
        
        # インデックス状態表示
        st.divider()
        st.subheader("📊 インデックス状態")
        if st.session_state.index_count > 0:
            st.write(f"**チャンク数**: {st.session_state.index_count}")
            if st.session_state.index_last_built:
                st.write(f"**最終更新**: {st.session_state.index_last_built}")
        else:
            st.info("インデックスが構築されていません。再構築ボタンを押してください。")

        # 評価実行セクション
        st.divider()
        st.subheader("🧪 評価実行")
        st.caption("LLM as a Judgeで回答品質を評価します（コストがかかります）")

        if st.button("📊 評価を実行", type="secondary", disabled=st.session_state.hybrid_retriever is None):
            st.session_state.eval_running = True

        # 評価結果の表示
        if st.session_state.eval_results is not None:
            summary = st.session_state.eval_results['summary']
            if summary['mvp_passed']:
                st.success(f"✅ MVP合格 ({summary['average_score']:.1f}点)")
            else:
                st.warning(f"❌ MVP不合格 ({summary['average_score']:.1f}点)")
            st.write(f"合格: {summary['passed_questions']}/{summary['total_questions']}問")


def main():
    """メイン関数"""
    
    # 機密情報注意表示
    render_security_notice()
    
    # サイドバー
    render_sidebar()
    
    # タイトル
    st.title("🔒 AIセキュリティ運用手順書アシスタント")
    st.markdown("手順書を検索して、質問に回答します。")
    
    # インデックスが構築されていない場合は初期化を促す
    if st.session_state.hybrid_retriever is None:
        st.warning("⚠️ インデックスが構築されていません。サイドバーから「インデックス再構築」を実行してください。")

        # 自動初期化を試みる
        if st.button("自動初期化を試す"):
            with st.spinner("初期化中..."):
                result = initialize_system(
                    st.session_state.data_folder,
                    bm25_weight=st.session_state.bm25_weight,
                    vector_weight=st.session_state.vector_weight,
                    k=st.session_state.k
                )
                if result['index_count'] > 0:
                    st.session_state.vectorstore = result['vectorstore']
                    st.session_state.hybrid_retriever = result['hybrid_retriever']
                    st.session_state.chunks_metadata = result['chunks_metadata']
                    st.session_state.index_count = result['index_count']
                    st.session_state.index_last_built = result['index_last_built']
                    st.success(f"初期化完了: {result['index_count']}件のチャンクをインデックス化しました")
                    st.rerun()
                else:
                    st.error("データフォルダにファイルが見つかりませんでした。")
        return

    # 評価実行処理
    if st.session_state.eval_running:
        st.session_state.eval_running = False  # フラグをリセット

        with st.spinner("評価を実行中... これには数分かかる場合があります"):
            try:
                # 評価データセット読み込み
                eval_dataset = load_eval_dataset()

                # 評価実行
                eval_results = run_evaluation_suite(
                    eval_dataset=eval_dataset,
                    answer_generator_func=handle_query,
                    session_state=st.session_state,
                    model=DEFAULT_JUDGE_MODEL
                )

                # 結果を保存
                saved_path = save_evaluation_results(eval_results)

                # セッション状態に保存
                st.session_state.eval_results = eval_results

                # 成功メッセージ
                st.success(f"評価完了！結果を {saved_path} に保存しました")

                # サマリーを表示
                st.markdown(format_evaluation_summary(eval_results['summary']))

                # 詳細結果を表示
                with st.expander("📋 詳細結果を見る"):
                    for result in eval_results['results']:
                        q_id = result['question_id']
                        category = result['category']
                        question = result['question']
                        eval_data = result['evaluation']

                        # 問題ごとの結果
                        passed_icon = "✅" if eval_data['passed'] else "❌"
                        st.markdown(f"### {passed_icon} 問題 {q_id} ({category}) - {eval_data['average_score']:.1f}点")
                        st.markdown(f"**質問**: {question}")

                        # スコア表示
                        st.markdown("**スコア**:")
                        cols = st.columns(3)
                        scores = eval_data['scores']
                        for i, (criteria, score) in enumerate(scores.items()):
                            col_idx = i % 3
                            cols[col_idx].metric(criteria, f"{score}点")

                        # 総合コメント
                        st.markdown(f"**総合コメント**: {eval_data['overall_comment']}")
                        st.divider()

                st.rerun()

            except Exception as e:
                st.error(f"評価中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 会話履歴表示
    for message in st.session_state.messages:
        render_chat_message(
            role=message['role'],
            content=message['content'],
            citations=message.get('citations', [])
        )
        
        # 危険操作警告バナー（各メッセージごとに表示）
        if message.get('flags', {}).get('dangerous_operation', False):
            render_danger_banner()
    
    # 質問入力
    if prompt := st.chat_input("質問を入力してください"):
        # ユーザーメッセージを追加
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'citations': []
        })
        
        # 会話履歴を制限（直近5往復 = 10メッセージ）
        if len(st.session_state.messages) > MAX_CONVERSATION_HISTORY * 2:
            st.session_state.messages = st.session_state.messages[-MAX_CONVERSATION_HISTORY * 2:]
        
        # 質問処理
        try:
            with st.spinner("回答を生成中..."):
                result = handle_query(prompt, st.session_state)
                
                # アシスタントメッセージを追加（警告は既にapply_guardrailsで追加済み）
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'citations': result['citations'],
                    'flags': result['flags'],
                    'warning_reason': result.get('warning_reason')
                })
                
                # ログ記録
                log_query(
                    query=prompt,
                    search_results=result['citations'],
                    answer=result['answer'],
                    processing_time=result['processing_time'],
                    token_usage=result['token_usage'],
                    cost=result['cost'],
                    search_config={
                        'k': st.session_state.k,
                        'bm25_weight': st.session_state.bm25_weight,
                        'vector_weight': st.session_state.vector_weight
                    },
                    flags=result['flags'],
                    warning_reason=result.get('warning_reason'),
                    top_score=result.get('top_score', 0.0),
                    session_id=st.session_state.session_id
                )
                
                # レイテンシ警告（5秒超過）
                if result['processing_time'] > 5.0:
                    st.warning(f"⚠️ 処理時間が5秒を超過しました（{result['processing_time']:.1f}秒）")
        
        except IndexNotBuiltError:
            handle_index_not_built()
            # ユーザーメッセージを削除（エラー時は追加しない）
            st.session_state.messages.pop()
        except APIError as e:
            st.error(f"❌ {e}")
            # ユーザーメッセージを削除
            st.session_state.messages.pop()
        except Exception as e:
            st.error(f"❌ 予期しないエラーが発生しました: {e}")
            import traceback
            with st.expander("エラー詳細", expanded=False):
                st.code(traceback.format_exc())
            # ユーザーメッセージを削除
            st.session_state.messages.pop()
        
        # 画面を更新
        st.rerun()


if __name__ == "__main__":
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEYが設定されていません。.envファイルまたは環境変数で設定してください。")
        st.stop()
    
    main()

