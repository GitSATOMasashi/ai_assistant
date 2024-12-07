from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
import tiktoken
import datetime
import re

# 環境変数の読み込み
load_dotenv() # 環境変数を読み込む

app = FastAPI() # FastAPIのインスタンスを作成

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static") # 静的ファイルをマウント
templates = Jinja2Templates(directory="templates") # テンプレートを設定

# CORSの設定
app.add_middleware( # CORSのミドルウェアを追加
    CORSMiddleware, 
    allow_origins=["*"], # すべてのオリジンを許可
    allow_credentials=True, # クレデンシャルを許可
    allow_methods=["*"], # すべてのメソッドを許可
    allow_headers=["*"], # すべてのヘッダーを許可
)

# Difyの設定
DIFY_API_URL = "https://api.dify.ai/v1/chat-messages" # DifyのAPI URL
DIFY_KEYS = { # DifyのAPIキー
    'bot1': os.getenv('DIFY_BOT1_KEY', '').split('#')[0].strip(), # 環境変数からDIFY_BOT1_KEYを取得し、#で分割して先頭の部分を取り出す
    'bot2': os.getenv('DIFY_BOT2_KEY', '').split('#')[0].strip(), # 環境変数からDIFY_BOT2_KEYを取得し、#で分割して先頭の部分を取り出す
    'bot3': os.getenv('DIFY_BOT3_KEY', '').split('#')[0].strip(), # 環境変数からDIFY_BOT3_KEYを取得し、#で分割して先頭の部分を取り出す
    'bot4': os.getenv('DIFY_BOT4_KEY', '').split('#')[0].strip() # 環境変数からDIFY_BOT4_KEYを取得し、#で分割して先頭の部分を取り出す
}

# デバッグ用に値を確認
print("DIFY_KEYS values:") # デバッグ用のメッセージを出力
for key, value in DIFY_KEYS.items(): # DIFY_KEYSの各キーと値を出力
    print(f"{key}: {value}") # キーと値を出力

# トークン管理クラス
class TokenManager: # トークン管理クラス
    def __init__(self, daily_limit=30000): # 初期化
        self.daily_limit = daily_limit # トークン上限
        self.user_tokens = {} # ユーザーのトークン使用状況
        # Claudeのエンコーディングを使用
        self.encoding = tiktoken.get_encoding("cl100k_base") # Claudeのエンコーディングを使用
    
    def count_tokens(self, text: str) -> dict: # トークン数を計算
        try:
            # tiktokenでトークン数を計算
            tokens = self.encoding.encode(text) # トークン数を計算
            return {
                "tokens": len(tokens), # トークン数を返す
                "method": "tiktoken",  # 計算方法を示す
                "success": True # 成功を示す
            }
        except Exception as e: # エラーが発生した場合
            print(f"Token counting error: {e}") # エラーを出力
            # フォールバック: 簡易的な計算方法
            japanese_chars = len([c for c in text if '\u3000' <= c <= '\u9fff']) # 日本語文字の数を取得
            other_chars = len(text) - japanese_chars # その他の文字の数を取得
            return { # トークン数を返す
                "tokens": max(1, int(japanese_chars * 2.5 + other_chars * 0.5)), # トークン数を返す
                "method": "fallback",  # 計算方法を示す
                "success": False # 成功を示す
            }

    def check_and_update_tokens(self, user_id: str, text: str) -> tuple[bool, int]: # トークン制限のチェックと更新
        today = datetime.date.today() # 今日の日付を取得
        
        # ユーザーの今日のトークン使用状況を初期化/更新
        if user_id not in self.user_tokens or self.user_tokens[user_id]["date"] != today: # ユーザーの今日のトークン使用状況がない場合
            self.user_tokens[user_id] = {"date": today, "count": 0} # ユーザーの今日のトークン使用状況を初期化/更新
        
        # トークン数を計算
        tokens_needed = self.count_tokens(text) # トークン数を計算
        tokens_remaining = self.daily_limit - self.user_tokens[user_id]["count"] # 残りのトークン数を計算
        
        # トークン制限チェック
        if tokens_needed > tokens_remaining: # トークン数が残りのトークン数を超える場合
            return False, tokens_remaining # 失敗を示す
        
        # トークンを消費
        self.user_tokens[user_id]["count"] += tokens_needed # トークン数を更新
        return True, self.daily_limit - self.user_tokens[user_id]["count"] # 成功を示す

token_manager = TokenManager() # トークン管理クラスのインスタンスを作成

@app.get("/", response_class=HTMLResponse) # ルートパスのGETリクエストを処理
async def root(request: Request): # ルートパスのGETリクエストを処理
    return templates.TemplateResponse("index.html", {"request": request}) # テンプレートを返す

@app.post("/calculate_tokens") # トークン数を計算するPOSTリクエストを処理
async def calculate_tokens(request: dict): # トークン数を計算するPOSTリクエストを処理
    text = request.get("text", "") # テキストを取得
    user_id = request.get("user_id", "default_user") # ユーザーIDを取得
    
    # トークン数を計算（消費はしない）
    token_info = token_manager.count_tokens(text) # トークン数を計算
    
    return { # トークン数を返す
        "tokens": token_info["tokens"], # トークン数
        "method": token_info["method"], # 計算方法
        "success": token_info["success"], # 成功を示す
        "remaining_tokens": token_manager.daily_limit - token_manager.user_tokens.get(user_id, {"count": 0})["count"] # 残りのトークン数
    }

@app.post("/chat/{bot_id}") # チャットのPOSTリクエストを処理
async def chat(bot_id: str, request: dict): # チャットのPOSTリクエストを処理
    if bot_id not in DIFY_KEYS: # ボットIDが有効でない場合
        raise HTTPException(status_code=400, detail="Invalid bot ID") # 400エラーを返す
    
    # トークン制限のチェックと更新
    user_id = request.get("user_id", "default_user") # ユーザーIDを取得
    token_info = token_manager.count_tokens(request["message"]) # トークン数を計算
    
    # 現在の残りトークン数を取得
    current_tokens = token_manager.user_tokens.get(user_id, {"count": 0})["count"] # 現在の残りトークン数を取得
    remaining_tokens = token_manager.daily_limit - current_tokens # 残りのトークン数を計算
    
    # トークン制限チェック
    if token_info["tokens"] > remaining_tokens: # トークン数が残りのトークン数を超える場合
        raise HTTPException( # 429エラーを返す
            status_code=429,
            detail={
                "error": "トークン制限に達しました",
                "remaining_tokens": remaining_tokens
            }
        )
    
    # トークンを消費
    token_manager.user_tokens[user_id] = { # ユーザーの今日のトークン使用状況を更新
        "date": datetime.date.today(), # 今日の日付を更新
        "count": current_tokens + token_info["tokens"] # トークン数を更新
    }
    
    headers = {
        "Authorization": f"Bearer {DIFY_KEYS[bot_id]}",
        "Content-Type": "application/json"
    }
    
    # 基本のリクエストデータ
    data = { # 基本のリクエストデータ
        "inputs": {}, # 入力
        "query": request["message"], # メッセージ
        "response_mode": "blocking", # レスポンスモード
        "user": request.get("user_id", "default_user") # ユーザーID
    }
    
    # 継続会話の場合のみconversation_idを含める
    if "conversation_id" in request and request["conversation_id"]: # 継続会話の場合
        data["conversation_id"] = request["conversation_id"] # conversation_idを含める
        print(f"Continuing conversation: {request['conversation_id']}") # デバッグ用のメッセージを出力
    else:
        print("Starting new conversation") # デバッグ用のメッセージを出力

    try:
        async with httpx.AsyncClient(timeout=120.0) as client: # リクエストを送信
            response = await client.post( # リクエストを送信
                "https://api.dify.ai/v1/chat-messages", # DifyのAPI URL
                headers=headers, # ヘッダー
                json=data # リクエストデータ
            )
            response.raise_for_status() # レスポンスをチェック
            response_data = response.json() # レスポンスをJSON形式に変換
            
            # 会話IDをログに出力（デバッグ用）
            if 'conversation_id' in response_data: # 会話IDがある場合
                print(f"Response conversation_id: {response_data['conversation_id']}") # デバッグ用のメッセージを出力
                
            return response_data # レスポンスを返す
            
    except UnicodeEncodeError as e: # 文字エンコーディングエラーが発生した場合
        print(f"UnicodeEncodeError details: {str(e)}") # デバッグ用のメッセージを出力
        print(f"Error position: {e.start} to {e.end}") # デバッグ用のメッセージを出力
        print(f"Object causing error: {e.object[e.start:e.end]}") # デバッグ用のメッセージを出力
        print(f"Full object: {e.object}") # デバッグ用のメッセージを出力
        raise HTTPException( # 500エラーを返す
            status_code=500,
            detail={"error": str(e), "message": "文字エンコーディングエラーが発生しました"}
        )
    except Exception as e: # 予期しないエラーが発生した場合
        print(f"Unexpected error: {type(e).__name__}") # デバッグ用のメッセージを出力
        print(f"Error details: {str(e)}") # デバッグ用のメッセージを出力
        raise HTTPException( # 500エラーを返す
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__": # メイン関数
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) # アプリケーションを起動