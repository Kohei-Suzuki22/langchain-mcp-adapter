# =============================================================================
# 標準ライブラリのインポート
# =============================================================================
import asyncio  # 非同期処理を行うためのライブラリ（async/awaitを使うために必要）
import os


# =============================================================================
# サードパーティライブラリのインポート
# =============================================================================
from dotenv import load_dotenv  # .envファイルから環境変数を読み込むためのライブラリ

# LangChain関連
from langchain_core.messages import HumanMessage  # ユーザーからのメッセージを表すクラス
from langchain_anthropic import ChatAnthropic  # Claude APIを使うためのLangChainラッパー
from langchain_mcp_adapters.tools import load_mcp_tools  # MCPサーバーのツールをLangChainで使えるようにする
from langgraph.prebuilt import create_react_agent  # ReActエージェントを簡単に作成するためのヘルパー

# MCP (Model Context Protocol) 関連
from mcp import ClientSession, StdioServerParameters  # MCPクライアントセッションと標準入出力のパラメータ
from mcp.client.stdio import stdio_client  # 標準入出力経由でMCPサーバーと通信するクライアント


# =============================================================================
# 初期設定
# =============================================================================

# .envファイルから環境変数を読み込む
# これにより ANTHROPIC_API_KEY などが os.environ から取得できるようになる
load_dotenv()


# Claude (Anthropic) のLLMインスタンスを作成
# model: 使用するClaudeのモデル名を指定
llm = ChatAnthropic(model="claude-sonnet-4-20250514")


# MCPサーバーへの接続設定（標準入出力方式）
# command: 実行するコマンド（ここではpython）
# args: コマンドに渡す引数（MCPサーバーのスクリプトパス）
# → これにより「python math_server.py」が実行され、標準入出力でMCPプロトコル通信する
stdio_server_params = StdioServerParameters(
    command="python",
    args=["/Users/kohei/source-code/ai-practice/langchain-mcp-adapter/servers/math_server.py"],
)


# =============================================================================
# メイン処理
# =============================================================================

async def main():
  """
  メインの非同期関数
  MCPサーバーに接続し、エージェントを使って質問に回答する
  """
  print("Hello from langchain-mcp-adapter!")

  # =========================================================================
  # Step 1: MCPサーバーとの接続を確立
  # =========================================================================
  # stdio_client: 標準入出力を使ってMCPサーバー（math_server.py）を起動し接続
  # read, write: サーバーとの通信用ストリーム（読み取り用、書き込み用）
  # async with: 処理が終わったら自動的に接続を閉じる（コンテキストマネージャー）
  async with stdio_client(stdio_server_params) as (read, write):

    # =========================================================================
    # Step 2: MCPセッションを作成・初期化
    # =========================================================================
    # ClientSession: MCPプロトコルでサーバーとやり取りするためのセッション
    # read_stream, write_stream: Step1で取得した通信ストリームを渡す
    async with ClientSession(read_stream=read, write_stream=write) as session:
      # セッションの初期化（MCPプロトコルのハンドシェイク）
      await session.initialize()
      print("session initialized")

      # =========================================================================
      # Step 3: MCPサーバーのツールをLangChain形式で取得
      # =========================================================================
      # load_mcp_tools(): MCPサーバーのツールをLangChainで使える形式に変換して取得
      # ※ session.list_tools()はMCP形式のまま返すので、LangChainでは使えない
      tools = await load_mcp_tools(session)
      print(f"利用可能なツール: {[tool.name for tool in tools]}")

      # =========================================================================
      # Step 4: ReActエージェントを作成
      # =========================================================================
      # create_react_agent: LLM（Claude）とツールを組み合わせてエージェントを作成
      # ReActパターン: Reasoning（推論）とAction（行動）を繰り返すAIエージェントの設計パターン
      # → LLMが「どのツールを使うか」を考え、ツールを実行し、結果を見て次の行動を決める
      agent = create_react_agent(llm, tools)

      # =========================================================================
      # Step 5: エージェントに質問を投げて回答を取得
      # =========================================================================
      # ainvoke: 非同期でエージェントを実行（aはasyncの略）
      # messages: チャット形式のメッセージリストを渡す
      # HumanMessage: ユーザーからの質問を表すメッセージオブジェクト
      result = await agent.ainvoke({"messages": [HumanMessage(content="What is 2 + 2?")]})

      # 結果のメッセージ一覧から最後のメッセージ（AIの回答）を取り出して表示
      print(result["messages"][-1].content)


# =============================================================================
# エントリーポイント
# =============================================================================

# このファイルが直接実行された場合のみ main() を実行する
# （他のファイルからimportされた場合は実行しない）
if __name__ == "__main__":
    # asyncio.run() で非同期関数 main() を実行する
    # これがPythonで非同期処理を開始する標準的な方法
    asyncio.run(main())
