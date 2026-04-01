import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP AWAL
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# NAMA FOLDER PENYIMPANAN DATABASE (LOKAL)
FOLDER_DB = "sakti_db_vektor"

# ==========================================
# 2. LOGIKA PENYIMPANAN & PEMUATAN DATABASE
# ==========================================
if os.path.exists(FOLDER_DB):
    # JIKA DATABASE SUDAH ADA: Langsung load dari folder (Super Cepat!)
    print("⚡ Membangunkan SAKTI... (Meload database lokal)")
    # Catatan: allow_dangerous_deserialization wajib True di versi LangChain terbaru untuk baca file lokal
    db_vektor = FAISS.load_local(FOLDER_DB, embeddings, allow_dangerous_deserialization=True)
    print("✅ Ingatan SAKTI berhasil dimuat dalam hitungan milidetik!")
else:
    # JIKA DATABASE BELUM ADA: Baca Word, Potong, Embed, lalu SIMPAN!
    print("📚 Membaca dokumen baru dan membangun ingatan SAKTI dari nol...")
    loader = Docx2txtLoader("data_pengetahuan/pedoman.docx")
    potongan_dokumen = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
    
    db_vektor = FAISS.from_documents(potongan_dokumen, embeddings)
    
    # Simpan ke folder lokal agar tidak amnesia
    db_vektor.save_local(FOLDER_DB)
    print(f"💾 Sempurna! Otak SAKTI berhasil disimpan permanen di folder '{FOLDER_DB}'.")

# ==========================================
# 3. MERAKIT RAG CHAIN
# ==========================================
pencari_konteks = db_vektor.as_retriever(search_kwargs={"k": 3}) # K=3 agar lebih banyak konteks yang diambil
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

template_sakti = """
Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip yang ramah dan solutif.
Jawab pertanyaan mahasiswa HANYA berdasarkan konteks dokumen di bawah ini.
Jika tidak ada jawabannya di konteks, katakan dengan jujur bahwa kamu belum tahu dan arahkan ke pengurus.

Konteks Dokumen:
{context}

Pertanyaan Mahasiswa: {question}

Jawaban SAKTI:"""
prompt = PromptTemplate.from_template(template_sakti)

rag_chain = (
    {"context": pencari_konteks, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. TES UJI COBA DATA BARU!
# ==========================================
# Ganti pertanyaan ini dengan salah satu kasus ekstrem yang baru saja kamu tambahkan di Word!
pertanyaan_mhs = "apa itu kipk?"

print("\n" + "="*50)
print(f"🙋‍♂️ Mahasiswa : {pertanyaan_mhs}")
print("🤖 SAKTI sedang mencari jawaban...")
print("="*50 + "\n")

jawaban_final = rag_chain.invoke(pertanyaan_mhs)
print(f"✨ SAKTI :\n{jawaban_final}\n")