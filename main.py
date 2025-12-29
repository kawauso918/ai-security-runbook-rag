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
    DEFAULT_LLM_MODEL, MAX_CONVERSATION_HISTORY
)
from initialize import initialize_system
from retriever import search_with_scores, update_retriever_weights, update_retriever_k
from guardrails import apply_guardrails
from logger import log_query
from components import (
    render_citation, render_danger_banner, render_security_notice,
    render_chat_message
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


def handle_query(user_query: str, session_state: Dict) -> Dict:
    """質問処理のオーケストレーション"""
    start_time = time.time()

    # Retrieverの重みとkを更新
    if session_state['hybrid_retriever'] is not None:
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
    search_results = search_with_scores(
        ensemble_retriever=session_state['hybrid_retriever'],
        query=user_query,
        k=session_state['k']
    )
    
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
    
    # プロンプト実行
    prompt = prompt_template.format_messages(
        context=context_text,
        history=history_text,
        question=user_query
    )
    
    response = llm.invoke(prompt)
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
            with st.spinner("インデックスを構築中..."):
                result = initialize_system(
                    st.session_state.data_folder,
                    bm25_weight=st.session_state.bm25_weight,
                    vector_weight=st.session_state.vector_weight,
                    k=st.session_state.k
                )
                st.session_state.vectorstore = result['vectorstore']
                st.session_state.hybrid_retriever = result['hybrid_retriever']
                st.session_state.chunks_metadata = result['chunks_metadata']
                st.session_state.index_count = result['index_count']
                st.session_state.index_last_built = result['index_last_built']
                st.success(f"インデックス構築完了: {result['index_count']}件")
        
        # インデックス状態表示
        st.divider()
        st.subheader("📊 インデックス状態")
        if st.session_state.index_count > 0:
            st.write(f"**チャンク数**: {st.session_state.index_count}")
            if st.session_state.index_last_built:
                st.write(f"**最終更新**: {st.session_state.index_last_built}")
        else:
            st.info("インデックスが構築されていません。再構築ボタンを押してください。")


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
        
        # 質問処理
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
        
        # 画面を更新
        st.rerun()


if __name__ == "__main__":
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEYが設定されていません。.envファイルまたは環境変数で設定してください。")
        st.stop()
    
    main()

