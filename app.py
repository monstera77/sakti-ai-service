import os
import ast
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakti_db_vektor")

SYSTEM_PROMPT = """Kamu adalah Sakabot, asisten virtual layanan informasi KIP Kuliah Universitas Diponegoro yang ramah dan solutif.
Tugas utamamu HANYA melayani informasi birokrasi dan kendala KIP Kuliah Undip.

Konteks Referensi KIPK:
{context}

Aturan (wajib dipatuhi):
1. Tolak pertanyaan di luar topik KIP Kuliah/Undip. Balas dengan: "Mohon maaf, Sakabot hanya diprogram untuk membantu layanan dan informasi seputar KIP Kuliah Undip."
2. Jika gambar dilampirkan, periksa relevansinya. Analisis hanya jika berupa screenshot portal Undip, dokumen administrasi, atau bukti error. Tolak gambar yang tidak relevan.
3. Jawab pertanyaan valid HANYA berdasarkan konteks referensi di atas. Jangan mengarang informasi.
4. Jika ditanya soal kemampuan memproses gambar, jawab dengan percaya diri bahwa kamu bisa, selama gambar relevan dengan KIP Kuliah."""

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="SAKTI AI Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
# Memuat brankas FAISS
vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
# Menyiapkan retriever utama (mengambil 3 dokumen)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    pesan: str
    gambar_base64: Optional[str] = None
    history: List[Dict[str, str]] = []

class TitleRequest(BaseModel):
    pesan: str

# ---------------------------------------------------------------------------
# Helpers & Semantic Guardrail
# ---------------------------------------------------------------------------

def is_semantically_valid(query: str) -> bool:
    """
    Common Sense Guardrail: Mengecek jarak makna (vektor) pertanyaan terhadap dokumen FAISS.
    """
    # 1. Tolak pesan yang terlalu pendek
    if len(query.strip()) < 5:
        print("🛑 [BLOCKED] Kueri terlalu pendek.")
        return False
        
    # 2. Cek kemiripan semantik dengan database (Ambil 1 dokumen teratas)
    docs_and_scores = vector_store.similarity_search_with_score(query, k=1)
    
    if not docs_and_scores:
        return False
        
    best_doc, score = docs_and_scores[0]
    
    # 3. Penetapan Batas Toleransi Jarak (Threshold L2 Distance)
    # Semakin kecil angka score, semakin relevan maknanya.
    # Anda bisa mengatur angka ini saat pengujian. Nilai umum FAISS berkisar 0.6 - 1.2
    THRESHOLD_JARAK = 0.95 
    
    if score > THRESHOLD_JARAK:
        print(f"🛑 [BLOCKED by Guardrail] OOT/Jailbreak terdeteksi. Skor Jarak: {score:.4f}")
        return False
        
    print(f"✅ [PASSED Guardrail] Kueri relevan. Skor Jarak: {score:.4f}")
    return True


def retrieve_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

def normalize_image_data(raw: str) -> str:
    if not raw.startswith("data:image"):
        return f"data:image/jpeg;base64,{raw}"
    return raw

def extract_text(response_content) -> str:
    if not isinstance(response_content, str):
        return response_content
    if response_content.strip().startswith("[{"):
        try:
            parsed = ast.literal_eval(response_content)
            if isinstance(parsed, list) and parsed:
                return parsed[0].get("text", response_content)
        except Exception:
            pass
    return response_content

def build_messages(query: str, context: str, image_data: Optional[str], history: list) -> list:
    system = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
    messages = [system]

    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role in ["user", "human"]:
            messages.append(HumanMessage(content=content))
        elif role in ["model", "assistant", "ai"]:
            messages.append(AIMessage(content=content))

    user_content = [{"type": "text", "text": query}]
    if image_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": normalize_image_data(image_data)},
        })
    
    messages.append(HumanMessage(content=user_content))
    return messages

def generate_short_title(query: str) -> str:
    prompt = (
        f"Buat judul singkat dan profesional (maksimal 4 kata) "
        f"yang merangkum inti dari pesan ini: '{query}'. "
        f"Jangan gunakan tanda kutip, langsung tuliskan judulnya."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().replace('"', '').replace("'", "")
    except Exception as e:
        print(f"⚠️ Error saat membuat judul: {e}")
        return "Percakapan Baru"

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "SAKTI AI Microservice is running."}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # =====================================================================
    # 1. SEMANTIC GUARDRAIL (Lapisan Deterministik / Common Sense)
    # =====================================================================
    if not is_semantically_valid(request.pesan):
         return {
             "status": "sukses", 
             "jawaban": "Sistem mendeteksi pertanyaan di luar cakupan atau tidak relevan. Mohon tanyakan secara spesifik seputar birokrasi KIP Kuliah Undip."
         }

    # =====================================================================
    # 2. RAG PIPELINE & LLM GENERATION (Hanya dieksekusi jika lolos Guardrail)
    # =====================================================================
    context = retrieve_context(request.pesan)
    messages = build_messages(request.pesan, context, request.gambar_base64, request.history)

    try:
        response = llm.invoke(messages)
        answer = extract_text(response.content)
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            print("⚠️ Terkena Limit API Menitan (Rate Limit)!")
            answer = "Mohon maaf, antrean API sedang penuh. Silakan tunggu 10 detik lalu coba lagi."
        else:
            print(f"⚠️ Terjadi error sistem: {e}")
            answer = "Mohon maaf, sistem sedang mengalami gangguan teknis."

    return {"status": "sukses", "jawaban": answer}

@app.post("/api/generate-title")
async def get_title(request: TitleRequest):
    judul = generate_short_title(request.pesan)
    return {"status": "sukses", "judul": judul}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)