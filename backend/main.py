import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent   
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)   # 确保目录存在
if not (FRONTEND_DIR / "index.html").exists():
    (FRONTEND_DIR / "index.html").write_text("<h1>ChatAgent API is running</h1>", encoding="utf-8")



load_dotenv()
app = FastAPI(title="RAG+Agent 简易对话服务", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 🔧 配置
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 🤖 模型与向量库初始化
llm = ChatOpenAI(model="qwen-plus", openai_api_key=API_KEY, openai_api_base=BASE_URL, temperature=0.7)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=API_KEY  
)

vector_store = FAISS.from_texts(["初始空知识库"], embeddings)
os.makedirs("./uploaded_docs", exist_ok=True)

# 🛠️ Agent 工具定义
@tool
def search_knowledge(query: str) -> str:
    """搜索已上传的文档内容。"""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "未找到相关文档内容。"
    return "\n---\n".join([
        f"📄 {d.metadata.get('source', '未知')}\n{d.page_content[:300]}..."
        for d in docs
    ])

@tool
def get_current_time(query: str = "") -> str:
    """获取当前系统时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_knowledge, get_current_time]

# 🧠 Agent 组装（新版：直接使用 LangGraph 预构建 ReAct Agent）
# 注意：新版本 create_react_agent 已经内置了提示模板，无需额外定义
# 如需自定义提示，可通过 prompt 参数传入，但格式需为 SystemMessage + MessagesPlaceholder 等
system_prompt = (
    "你是一个智能助手。你可以使用提供的工具回答问题。"
    "当用户询问文档内容时，务必使用 search_knowledge 工具。"
    "回答请保持简洁专业，优先使用中文。"
)

agent = create_agent(
    model=llm,          
    tools=tools,
    system_prompt=system_prompt   
)

sessions = {}

# 📤 文件上传接口（自动分块+向量化）
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "请上传 .txt 或 .md 纯文本文件（PDF/Word需额外安装解析库）")

    fname = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(f"./uploaded_docs/{fname}", "w", encoding="utf-8") as f:
        f.write(text)

    doc = Document(page_content=text, metadata={"source": file.filename})
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents([doc])
    vector_store.add_documents(chunks)
    return {"status": "success", "message": f"✅ 已处理并索引: {file.filename}"}

# 💬 对话接口（Agent 驱动）
@app.post("/api/chat")
async def chat(payload: dict):
    msg = payload.get("message", "").strip()
    sid = payload.get("session_id", "default")
    if not msg:
        raise HTTPException(400, "消息不能为空")

    # 获取会话历史（旧格式：{"role":..., "content":...}）
    history = sessions.get(sid, [])

    # 转换为 LangGraph 期望的消息列表
    lc_messages = []
    for turn in history:
        if turn["role"] == "user":
            lc_messages.append(HumanMessage(content=turn["content"]))
        else:
            lc_messages.append(AIMessage(content=turn["content"]))

    # 追加当前用户消息
    lc_messages.append(HumanMessage(content=msg))

    try:
        # 新版调用方式：传入 {"messages": [...]}，返回的最后一个消息即为助手回复
        result = await agent.ainvoke({"messages": lc_messages})
        # 从返回的消息列表中提取最后一条 AIMessage 的内容
        reply = result["messages"][-1].content
    except Exception as e:
        raise HTTPException(500, f"Agent 执行异常: {str(e)}")

    # 更新会话历史
    history = history[-8:] + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": reply}
    ]
    sessions[sid] = history

    return {"reply": reply}

# 🌐 静态网页挂载
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



