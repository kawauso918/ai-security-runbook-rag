"""UIコンポーネント（再利用可能な部品）"""

import streamlit as st
from typing import List, Dict


def render_citation(citation: Dict) -> str:
    """引用表示コンポーネント（Markdown形式）"""
    file = citation.get('file', '')
    heading = citation.get('heading', '')
    score = citation.get('score', 0.0)
    text = citation.get('text', '')
    
    # ファイル名のみ取得
    file_name = file.split('/')[-1] if '/' in file else file
    
    # 抜粋テキスト（最大100文字）
    excerpt = text[:100] + '...' if len(text) > 100 else text
    
    markdown = f"""<details>
<summary>引用元: {file_name} {'> ' + heading if heading else ''} (スコア: {score:.2f})</summary>

{excerpt}
</details>"""
    
    return markdown


def render_danger_banner() -> None:
    """危険操作警告バナー表示"""
    st.warning("⚠️ 承認・確認が必要", icon="⚠️")


def render_security_notice() -> None:
    """機密情報入力禁止の注意表示"""
    st.info("⚠️ 機密情報の入力は禁止されています", icon="🔒")


def render_chat_message(role: str, content: str, citations: List[Dict] = None) -> None:
    """チャットメッセージ表示"""
    if role == 'user':
        with st.chat_message("user"):
            st.write(content)
    elif role == 'assistant':
        with st.chat_message("assistant"):
            st.markdown(content)
            
            # 引用を表示
            if citations:
                st.markdown("### 引用元")
                for citation in citations:
                    st.markdown(render_citation(citation), unsafe_allow_html=True)

