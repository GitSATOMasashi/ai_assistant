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
load_dotenv()

app = FastAPI()

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Difyの設定
DIFY_API_URL = "https://api.dify.ai/v1/chat-messages"
DIFY_KEYS = {
    'bot1': os.getenv('DIFY_BOT1_KEY'),
    'bot2': os.getenv('DIFY_BOT2_KEY'),
    'bot3': os.getenv('DIFY_BOT3_KEY'),
    'bot4': os.getenv('DIFY_BOT4_KEY')
}

# トークン管理クラス
class TokenManager:
    def __init__(self, daily_limit=30000):
        self.daily_limit = daily_limit
        self.user_tokens = {}
        # Claudeのエンコーディングを使用
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> dict:
        try:
            # tiktokenでトークン数を計算
            tokens = self.encoding.encode(text)
            return {
                "tokens": len(tokens),
                "method": "tiktoken",  # 計算方法を示す
                "success": True
            }
        except Exception as e:
            print(f"Token counting error: {e}")
            # フォールバック: 簡易的な計算方法
            japanese_chars = len([c for c in text if '\u3000' <= c <= '\u9fff'])
            other_chars = len(text) - japanese_chars
            return {
                "tokens": max(1, int(japanese_chars * 2.5 + other_chars * 0.5)),
                "method": "fallback",  # 計算方法を示す
                "success": False
            }

    def check_and_update_tokens(self, user_id: str, text: str) -> tuple[bool, int]:
        today = datetime.date.today()
        
        # ユーザーの今日のトークン使用状況を初期化/更新
        if user_id not in self.user_tokens or self.user_tokens[user_id]["date"] != today:
            self.user_tokens[user_id] = {"date": today, "count": 0}
        
        # トークン数を計算
        tokens_needed = self.count_tokens(text)
        tokens_remaining = self.daily_limit - self.user_tokens[user_id]["count"]
        
        # トークン制限チェック
        if tokens_needed > tokens_remaining:
            return False, tokens_remaining
        
        # トークンを消費
        self.user_tokens[user_id]["count"] += tokens_needed
        return True, self.daily_limit - self.user_tokens[user_id]["count"]

token_manager = TokenManager()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate_tokens")
async def calculate_tokens(request: dict):
    text = request.get("text", "")
    user_id = request.get("user_id", "default_user")
    
    # トークン数を計算（消費はしない）
    token_info = token_manager.count_tokens(text)
    
    return {
        "tokens": token_info["tokens"],
        "method": token_info["method"],
        "success": token_info["success"],
        "remaining_tokens": token_manager.daily_limit - token_manager.user_tokens.get(user_id, {"count": 0})["count"]
    }

@app.post("/chat/{bot_id}")
async def chat(bot_id: str, request: dict):
    if bot_id not in DIFY_KEYS:
        raise HTTPException(status_code=400, detail="Invalid bot ID")
    
    headers = {
        "Authorization": f"Bearer {DIFY_KEYS[bot_id]}",
        "Content-Type": "application/json"
    }
    
    # 基本のリクエストデータ
    data = {
        "inputs": {},
        "query": request["message"],
        "response_mode": "blocking",
        "user": request.get("user_id", "default_user")
    }
    
    # 継続会話の場合のみconversation_idを含める
    if "conversation_id" in request and request["conversation_id"]:
        data["conversation_id"] = request["conversation_id"]
        print(f"Continuing conversation: {request['conversation_id']}")
    else:
        print("Starting new conversation")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.dify.ai/v1/chat-messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            response_data = response.json()
            
            # レスポンスの整形
            if 'answer' in response_data:
                # HTMLの改行タグに変換
                answer = response_data['answer'].replace('\n', '<br>')
                response_data['answer'] = answer
            
            # 会話IDをログに出力（デバッグ用）
            if 'conversation_id' in response_data:
                print(f"Response conversation_id: {response_data['conversation_id']}")
                
            return response_data
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 