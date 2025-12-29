"""エラーハンドリング処理"""

from typing import Optional, Tuple, List
import streamlit as st


class DataFolderEmptyError(Exception):
    """データフォルダが空の場合のエラー"""
    pass


class PDFReadError(Exception):
    """PDFファイルが読めない場合のエラー"""
    pass


class IndexNotBuiltError(Exception):
    """インデックスが構築されていない場合のエラー"""
    pass


class APIError(Exception):
    """API呼び出しエラー"""
    pass


def handle_data_folder_empty(data_folder: str) -> None:
    """データフォルダが空の場合の処理"""
    st.error(f"❌ データフォルダ `{data_folder}` にファイルが見つかりません。")
    st.info("📝 以下の手順でデータを追加してください：\n"
            "1. `data/` フォルダに手順書ファイル（.md, .txt, .pdf）を配置\n"
            "2. サイドバーから「インデックス再構築」を実行")


def handle_pdf_read_error(file_path: str, error: Exception) -> None:
    """PDFファイルが読めない場合の処理"""
    st.warning(f"⚠️ PDFファイル `{file_path}` の読み込みに失敗しました: {error}")
    st.info("💡 対応方法：\n"
            "- PDFファイルが破損していないか確認\n"
            "- パスワード保護されていないか確認\n"
            "- 他のファイル形式（.md, .txt）に変換して再試行")


def handle_index_not_built() -> None:
    """インデックスが構築されていない場合の処理"""
    st.warning("⚠️ インデックスが構築されていません。")
    st.info("📝 以下の手順でインデックスを構築してください：\n"
            "1. `data/` フォルダに手順書ファイルを配置\n"
            "2. サイドバーから「🔄 インデックス再構築」ボタンをクリック")


def handle_api_error(error: Exception, retry_count: int = 0) -> Tuple[bool, str]:
    """API呼び出しエラーの処理
    
    Returns:
        (should_retry, error_message) のタプル
    """
    error_msg = str(error)
    
    # リトライ可能なエラー
    retryable_errors = [
        "rate limit",
        "timeout",
        "connection",
        "503",
        "502",
        "500"
    ]
    
    should_retry = any(keyword in error_msg.lower() for keyword in retryable_errors) and retry_count < 3
    
    if should_retry:
        return True, f"⚠️ API呼び出しエラー（リトライ可能）: {error_msg}\nリトライ中..."
    else:
        return False, f"❌ API呼び出しエラー: {error_msg}\n\n💡 対応方法：\n- APIキーが正しく設定されているか確認\n- ネットワーク接続を確認\n- しばらく待ってから再試行"


def display_error_summary(errors: list[tuple[str, Exception]]) -> None:
    """エラーサマリーを表示"""
    if errors:
        with st.expander("⚠️ エラー詳細", expanded=False):
            for file_path, error in errors:
                st.error(f"`{file_path}`: {error}")

