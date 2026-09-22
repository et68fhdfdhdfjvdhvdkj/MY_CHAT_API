import os
import socket
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field


def get_available_port(start_port: int = 8000, end_port: int = 8100) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


def use_mock_mode() -> bool:
    value = os.environ.get("USE_MOCK_MODE", "false").lower()
    return value in {"1", "true", "yes", "on"}


def get_allowed_origins() -> list[str]:
    raw_value = os.environ.get("CORS_ORIGINS")
    if not raw_value:
        return ["*"]

    origins = []
    for item in raw_value.replace(";", ",").split(","):
        origin = item.strip()
        if origin:
            origins.append(origin)
    return origins or ["*"]


def get_host() -> str:
    return os.environ.get("HOST", "0.0.0.0")


def get_port() -> int:
    value = os.environ.get("PORT", "8000")
    try:
        return int(value)
    except ValueError:
        return 8000


def generate_mock_reply(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return "Mock mode is active. Please send a real message."
    return (
        f"Offline mock response: I received your message: '{text}'. "
        "The model API is unavailable right now, so this is a fallback reply."
    )


app = FastAPI(title="Coding Tutor API - Together AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEFAULT_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.

Your job is to help the user with a broad range of tasks, including coding, debugging, writing, analysis, planning, explanations, brainstorms, and productivity support.

Core behavior:
- Be helpful, clear, and concise.
- Use simple language unless the user asks for technical depth.
- Keep context across the full conversation.
- Ask clarifying questions when needed.
- Prefer practical, complete responses.
- If the user asks for code, provide working code when possible.
- If the user asks for explanations, break them down clearly.
- Stay focused on the user's request and avoid unrelated topics.
- Be honest about uncertainty and explain tradeoffs when relevant.
"""


def get_system_instruction() -> str:
    return os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


def build_messages(request: ChatRequest) -> list[dict[str, str]]:
    history = [
        item
        for item in request.history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    return [
        {"role": "system", "content": get_system_instruction()},
        *history,
        {"role": "user", "content": request.message},
    ]


def get_api_key() -> str:
    token = os.environ.get("TOGETHER_AI_API_KEY")
    if token and token.strip():
        return token.strip()
    return ""


def get_model_client() -> OpenAI:
    token = get_api_key()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="No API token found. Set TOGETHER_AI_API_KEY.",
        )

    return OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=token
    )


async def stream_mock_reply(message: str) -> AsyncIterator[str]:
    reply = generate_mock_reply(message)
    for word in reply.split(" "):
        yield f"data: {word} \n\n"
    yield "data: [DONE]\n\n"


def stream_model_reply(request: ChatRequest) -> AsyncIterator[str]:
    client = get_model_client()
    response = client.chat.completions.create(
        model=os.environ.get("TOGETHER_AI_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
        messages=build_messages(request),
        stream=True,
    )

    async def chunks() -> AsyncIterator[str]:
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return chunks()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server processing error: {str(exc)}"},
    )


@app.get("/")
def health_check():
    return {
        "status": "Server active with Together AI",
        "mock_mode": use_mock_mode(),
        "token_configured": bool(get_api_key()),
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(CHAT_PAGE)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/test")
def test_route():
    return {
        "message": "API is working",
        "mock_mode": use_mock_mode(),
        "token_configured": bool(get_api_key()),
    }


@app.get("/api/config")
def config_route():
    return {
        "title": app.title,
        "model": os.environ.get("TOGETHER_AI_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
        "mock_mode": use_mock_mode(),
        "token_configured": bool(get_api_key()),
        "cors_origins": get_allowed_origins(),
        "host": get_host(),
        "port": get_port(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if use_mock_mode() or not get_api_key():
        return ChatResponse(reply=generate_mock_reply(request.message))

    try:
        client = get_model_client()
        response = client.chat.completions.create(
            model=os.environ.get("TOGETHER_AI_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
            messages=build_messages(request),
        )

        content = response.choices[0].message.content
        if content is None:
            content = "No response was returned by the model."

        return ChatResponse(reply=content)

    except Exception as exc:
        return ChatResponse(
            reply=f"Offline mock response: Connection to model provider failed: {exc}. This is a fallback reply."
        )


@app.get("/api/chat")
def chat_endpoint_info():
    return {
        "message": "Chat endpoint is available.",
        "method": "POST",
        "body": {"message": "your coding question", "history": []},
        "streaming_endpoint": "/api/chat/stream",
    }


@app.post("/api/chat/stream")
async def stream_chat_endpoint(request: ChatRequest):
    if use_mock_mode() or not get_api_key():
        stream = stream_mock_reply(request.message)
    else:
        try:
            stream = stream_model_reply(request)
        except Exception as exc:
            async def error_stream() -> AsyncIterator[str]:
                yield f"data: Offline mock response: Connection failed: {exc}\n\n"
                yield "data: [DONE]\n\n"

            stream = error_stream()

    return StreamingResponse(stream, media_type="text/event-stream")


CHAT_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coding Tutor</title>
<style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;font:15px system-ui,-apple-system,sans-serif;color:#202124;background:#fff}
.app{display:grid;grid-template-columns:260px 1fr;min-height:100vh}.sidebar{padding:24px 18px;background:#f7f8fc;border-right:1px solid #e8eaf0}
.brand{display:flex;align-items:center;gap:10px;font-size:20px;font-weight:600;color:#3c4043}.spark{color:#6750a4;font-size:25px}.new-chat,.games-link{width:100%;border:0;border-radius:22px;padding:12px 16px;background:#e8def8;color:#3c246b;text-align:left;font:inherit;cursor:pointer}.new-chat{margin:28px 0 10px}.games-link{background:transparent;color:#4f5357}
.tip{color:#6b7075;line-height:1.5;font-size:13px}.content{display:flex;flex-direction:column;min-width:0}.topbar{display:flex;justify-content:space-between;align-items:center;padding:18px 32px;color:#5f6368}.model{font-size:18px;color:#3c4043}.model span{color:#6750a4}.status{font-size:13px}.status::before{content:'';display:inline-block;width:8px;height:8px;margin-right:6px;border-radius:50%;background:#34a853}
.conversation{width:min(860px,100%);flex:1;margin:0 auto;padding:44px 28px 24px}.welcome{text-align:center;padding:55px 0 42px}.welcome h1{font-size:clamp(30px,5vw,48px);font-weight:500;margin:0 0 12px;background:linear-gradient(90deg,#4285f4,#9b72cb,#d96570);color:transparent;background-clip:text}.welcome p{color:#70757a;margin:0}
#messages{display:grid;gap:18px}.message{max-width:82%;padding:14px 18px;border-radius:20px;white-space:pre-wrap;line-height:1.55}.user{justify-self:end;background:#f0eafa;color:#302044;border-bottom-right-radius:6px}.assistant{justify-self:start;background:#f8f9fa;border:1px solid #eceef2;border-bottom-left-radius:6px}
.composer{display:flex;align-items:flex-end;gap:10px;width:min(860px,100%);margin:0 auto;padding:14px 28px 25px}form{display:flex;align-items:flex-end;gap:10px;flex:1;padding:8px 10px 8px 16px;border:1px solid #dfe1e5;border-radius:28px;box-shadow:0 2px 8px #00000012;background:#fff}textarea{flex:1;min-height:24px;max-height:140px;resize:vertical;border:0;outline:0;padding:8px 0;font:inherit;color:#202124}button{border:0;cursor:pointer}form button{width:40px;height:40px;border-radius:50%;background:#6750a4;color:#fff;font-size:18px}form button:disabled{opacity:.5}.fine-print{text-align:center;color:#9aa0a6;font-size:11px;margin:-12px 0 10px}
.games{display:none;width:min(860px,100%);margin:0 auto;padding:28px}.games.active{display:block}.games h2{font-size:30px;font-weight:500;margin:10px 0 8px}.games-intro{color:#70757a;margin:0 0 24px}.game-card{border:1px solid #e4e5ea;border-radius:18px;padding:20px;background:#fafbff}.game-card h3{margin:0 0 8px}.game-card code{display:block;margin:14px 0;padding:16px;border-radius:12px;background:#202124;color:#e8eaed;font:14px monospace;white-space:pre-wrap}.answers{display:flex;gap:10px;flex-wrap:wrap}.answers button{padding:10px 16px;border-radius:18px;background:#eee8fb;color:#452c73}.answers button:hover{background:#ddcff4}.game-result{min-height:24px;font-weight:600}.back-chat{margin-top:22px;padding:10px 16px;border-radius:18px;background:#f0f1f5;color:#4b4f54}
@media(max-width:700px){.app{display:block}.sidebar{display:none}.topbar{padding:16px 20px}.conversation{padding:20px}.welcome{padding:45px 0 30px}.composer{padding:12px 14px 18px}.message{max-width:92%}}
</style></head>
<body><div class="app"><aside class="sidebar"><div class="brand"><span class="spark">✦</span>Coding Tutor</div><button class="new-chat" onclick="showChat()">＋ New chat</button><button class="games-link" onclick="showGames()">🎮 Games</button><p class="tip">Ask for explanations, debugging help, project ideas, or code reviews.</p></aside>
<section class="content"><header class="topbar"><div class="model">Coding Tutor <span>▾</span></div><div class="status">Online</div></header><div class="conversation" id="chat-view"><div class="welcome" id="welcome"><h1>How can I help you?</h1><p>Your focused coding companion.</p></div><div id="messages" aria-live="polite"></div></div>
<div class="composer"><form><textarea rows="1" placeholder="Ask anything about code..." required></textarea><button title="Send">↑</button></form></div><div class="fine-print">Coding Tutor can make mistakes. Check important code before using it.</div></section></div>
<section class="games" id="games-view"><h2>Take a coding break</h2><p class="games-intro">Practice small programming ideas with quick challenges.</p><div class="game-card"><h3>Guess the output</h3><p>What will this Python code print?</p><code>items = [1, 2, 3]
print(items[-1])</code><div class="answers"><button onclick="answerGame(this, '2')">2</button><button onclick="answerGame(this, '3')">3</button><button onclick="answerGame(this, '1')">1</button></div><p class="game-result" id="game-result" aria-live="polite"></p><button class="back-chat" onclick="showChat()">← Back to chat</button></div></section>
<script>
const messages=document.querySelector('#messages'), form=document.querySelector('form'), input=document.querySelector('textarea');
const history=[];
function showGames(){document.querySelector('#chat-view').style.display='none';document.querySelector('.composer').style.display='none';document.querySelector('.fine-print').style.display='none';document.querySelector('#games-view').classList.add('active')}
function showChat(){document.querySelector('#chat-view').style.display='block';document.querySelector('.composer').style.display='flex';document.querySelector('.fine-print').style.display='block';document.querySelector('#games-view').classList.remove('active')}
function answerGame(button, answer){const result=document.querySelector('#game-result');result.textContent=answer==='3'?'Correct! Python uses -1 for the last item.':'Not quite. The answer is 3.';result.style.color=answer==='3'?'#188038':'#b3261e'}
function add(role, text){const el=document.createElement('div');el.className='message '+role;el.textContent=text;messages.append(el);return el}
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener('submit', async event=>{event.preventDefault();const text=input.value.trim();if(!text)return;
document.querySelector('#welcome').style.display='none';
add('user',text);history.push({role:'user',content:text});input.value='';form.querySelector('button').disabled=true;
const output=add('assistant','');history.push({role:'assistant',content:''});
try{const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history:history.slice(0,-2)})});
if(!response.ok)throw new Error('Request returned '+response.status);const data=await response.json();output.textContent=data.reply;history.at(-1).content=data.reply;
}catch(error){output.textContent='Unable to reach the server: '+error.message;history.at(-1).content=output.textContent
}finally{form.querySelector('button').disabled=false;input.focus()}});
</script></body></html>"""
if __name__ == "__main__":
    host = get_host()
    port = get_port()
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=120)

