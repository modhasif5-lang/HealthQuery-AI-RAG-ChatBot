from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
import os
import json

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import the RAG logic from src
from src.chain_utils import run_elite_pipeline, stream_elite_pipeline

app = FastAPI(title="HealthQuery AI Medical RAG Chatbot")

# Rate Limiter: Limit requests per IP to protect Free Tier quotas
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/chat")
@limiter.limit("6/minute")
async def chat(request: Request, message: str = Form(...), session_id: str = Form("web_session_1")):
    try:
        response, intent = run_elite_pipeline(session_id, message)
        return JSONResponse(content={"response": response, "intent": intent})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat/stream")
@limiter.limit("6/minute")
async def chat_stream(request: Request, message: str = Form(...), session_id: str = Form("web_session_1")):
    """
    SSE streaming endpoint with Rate Limiting (6/min).
    """
    def event_generator():
        try:
            for item in stream_elite_pipeline(session_id, message):
                if len(item) == 3:
                    token, intent, status = item
                else:
                    token, intent = item
                    status = ""
                data = json.dumps({"token": token, "intent": intent, "status": status, "done": False})
                yield f"data: {data}\n\n"
            # Send completion signal
            yield f"data: {json.dumps({'token': '', 'intent': '', 'status': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Mount Gradio Blocks on /gradio so Hugging Face detects Gradio SDK cleanly
with gr.Blocks(title="HealthQuery AI Medical RAG") as demo:
    gr.Markdown("# 🩺 HealthQuery AI Medical RAG Chatbot\nVisit the [Interactive Chat Interface](/) for the full experience.")

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
