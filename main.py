import os
import ast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# --- IMPORT BARU UNTUK MULTIMODAL ---
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional

# 1. SETUP API & CORS
app = FastAPI(title="SAKTI AI Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD OTAK SAKTI & DATABASE
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

print("⚡ Menyalakan Server AI SAKTI (Multimodal Edition)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sakti_db_vektor") 

db_vektor = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
pencari_konteks = db_vektor.as_retriever(search_kwargs={"k": 3})

# Gunakan Gemini 2.5 Flash sesuai dengan model pilihan Anda
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 3. FORMAT DATA FRONTEND
class PesanMahasiswa(BaseModel):
    pesan: str
    gambar_base64: Optional[str] = None 

# 4. JEMBATAN API
@app.post("/api/chat")
async def chat_sakti(request: PesanMahasiswa):
    print(f"📥 Pertanyaan masuk: {request.pesan}")
    
    # --- PROSES RAG MULTIMODAL ---
    # Langkah A: Cari referensi di buku pedoman KIPK menggunakan TEKS saja
    dokumen_referensi = pencari_konteks.invoke(request.pesan)
    konteks_teks = "\n\n".join([doc.page_content for doc in dokumen_referensi])
    
    # Langkah B: Buat Karakter dan Instruksi SAKTI (Prompt yang sudah diperbarui)
    instruksi_sistem = f"""Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip yang ramah dan solutif.
    Kamu adalah AI cerdas yang MAMPU membaca dan menganalisis gambar/screenshot yang dilampirkan oleh mahasiswa.
    
    ATURAN MENJAWAB:
    1. Jika mahasiswa bertanya tentang aturan, syarat, atau informasi KIP Kuliah, jawab HANYA berdasarkan konteks dokumen berikut:
    {konteks_teks}
    2. Jika mahasiswa melampirkan gambar, perhatikan gambar tersebut baik-baik dan berikan solusi yang relevan secara logis.
    3. Jika ditanya apakah kamu bisa melihat/memproses gambar, jawablah dengan percaya diri bahwa kamu bisa!"""

    # Langkah C: Siapkan paket pesan dari mahasiswa
    konten_user = [{"type": "text", "text": request.pesan}]
    
    # Jika mahasiswa meng-upload gambar, masukkan ke dalam paket!
    if request.gambar_base64:
        print("📸 Menerima lampiran gambar dari mahasiswa!")
        img_data = request.gambar_base64
        # Pastikan formatnya terbaca oleh LangChain (wajib ada header data:image)
        if not img_data.startswith("data:image"):
            img_data = f"data:image/jpeg;base64,{img_data}"
            
        konten_user.append({
            "type": "image_url",
            "image_url": {"url": img_data}
        })
        
    # Langkah D: Satukan dan kirim ke Gemini
    paket_pesan = [
        SystemMessage(content=instruksi_sistem),
        HumanMessage(content=konten_user)
    ]
    
    respon_llm = llm.invoke(paket_pesan)
    jawaban = respon_llm.content
    
    # --- FILTER PEMBERSIH JSON (Jaga-jaga kalau Gemini kumat) ---
    if isinstance(jawaban, str) and jawaban.strip().startswith("[{"):
        try:
            parsed = ast.literal_eval(jawaban)
            if isinstance(parsed, list) and len(parsed) > 0:
                jawaban = parsed[0].get("text", jawaban)
        except Exception as e:
            print(f"⚠️ Gagal membersihkan format: {e}")
            pass
            
    print("📤 Mengirim balasan ke Frontend...")
    return {
        "status": "sukses",
        "jawaban": jawaban
    }

@app.get("/")
async def root():
    return {"message": "Server SAKTI AI Multimodal Berjalan Lancar! 🚀"}