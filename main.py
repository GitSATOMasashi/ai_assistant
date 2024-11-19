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
    'bot3': os.getenv('DIFY_BOT3_KEY')
}

# トークン管理クラス
class TokenManager:
    def __init__(self, daily_limit=2000):
        self.daily_limit = daily_limit
        self.user_tokens = {}  # {user_id: {"date": date, "count": count}}
    
    def count_tokens(self, text: str) -> int:
        # 日本語文字（漢字、ひらがな、カタカナ）を検出
        japanese_chars = len(re.findall(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3400-\u4dbf]', text))
        # 英数字と記号
        other_chars = len(re.findall(r'[a-zA-Z0-9\s\W]', text))
        
        # 日本語は1文字2.5トークン、その他は1文字0.5トークンとして計算
        total_tokens = int(japanese_chars * 2.5 + other_chars * 0.5)
        return max(1, total_tokens)  # 最低1トークンを保証
    
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
    user_id = request.get("user_id", "default_user")
    text = request.get("text", "")
    
    # トークンを計算して消費
    can_proceed, remaining_tokens = token_manager.check_and_update_tokens(user_id, text)
    if not can_proceed:
        raise HTTPException(
            status_code=429,
            detail={"error": "Daily token limit exceeded", "remaining_tokens": remaining_tokens}
        )
    
    return {"remaining_tokens": remaining_tokens}

@app.post("/chat/{bot_id}")
async def chat(bot_id: str, request: dict):
    if bot_id not in DIFY_KEYS:
        raise HTTPException(status_code=400, detail="Invalid bot ID")
    
    headers = {
        "Authorization": f"Bearer {DIFY_KEYS[bot_id]}",
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs": {},
        "query": request["message"],
        "response_mode": "blocking",
        "conversation_id": request.get("conversation_id"),
        "user": request.get("user_id", "default_user")
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.dify.ai/v1/chat-messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 