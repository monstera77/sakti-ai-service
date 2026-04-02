import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

# 1. SETUP API & CORS (Agar Next.js tidak diblokir saat nembak API)
app = FastAPI(title="SAKTI AI Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Dalam tahap dev, izinkan semua domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD OTAK SAKTI
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

print("⚡ Menyalakan Server AI SAKTI...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# Cukup gunakan nama foldernya langsung untuk Railway
db_vektor = FAISS.load_local("sakti_db_vektor", embeddings, allow_dangerous_deserialization=True)
pencari_konteks = db_vektor.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

template_sakti = """Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip yang ramah dan solutif.
Jawab pertanyaan HANYA berdasarkan konteks dokumen di bawah ini.
Konteks: {context}
Pertanyaan: {question}
Jawaban SAKTI:"""

prompt = PromptTemplate.from_template(template_sakti)
rag_chain = ({"context": pencari_konteks, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# 3. FORMAT DATA DARI FRONTEND
class PesanMahasiswa(BaseModel):
    pesan: str
    gambar_base64: Optional[str] = None # (Optional artinya boleh kosong kalau mhs cuma ngetik teks)

# 4. JEMBATAN API (ENDPOINT) UNTUK FRONTEND
@app.post("/api/chat")
async def chat_sakti(request: PesanMahasiswa):
    print(f"📥 Pertanyaan masuk dari Frontend: {request.pesan}")
    
    # Eksekusi RAG
    jawaban = rag_chain.invoke(request.pesan)
    
    print("📤 Mengirim balasan ke Frontend...")
    return {
        "status": "sukses",
        "jawaban": jawaban
    }

@app.get("/")
async def root():
    return {"message": "Server SAKTI AI Berjalan Lancar di Railway! 🚀"}