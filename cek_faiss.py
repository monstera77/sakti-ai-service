import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Muat API Key
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Siapkan Model Embedding (Harus sama persis dengan yang dipakai saat membuat DB)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

# 3. Buka Database FAISS
DB_PATH = "sakti_db_vektor"
print("⏳ Sedang membuka brankas FAISS...")
vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# ==========================================
# PENGECEKAN 1: Berapa banyak data di dalam?
# ==========================================
total_chunks = vector_store.index.ntotal
print(f"\n✅ DATABASE TERBACA!")
print(f"📊 Total potongan paragraf (chunks) yang tersimpan: {total_chunks} bagian")

# ==========================================
# PENGECEKAN 2: Mengintip salah satu teks yang tersimpan
# ==========================================
print("\n🔍 MENGINTIP ISI DOKUMEN PERTAMA:")
kamus_dokumen = vector_store.docstore._dict
id_dokumen_pertama = list(kamus_dokumen.keys())[0]
print("--------------------------------------------------")
print(kamus_dokumen[id_dokumen_pertama].page_content)
print("--------------------------------------------------")

# ==========================================
# PENGECEKAN 3: Tes Pencarian Murni (Tanpa Gemini)
# ==========================================
pertanyaan_tes = "Bagaimana tata cara pendaftaran KIP Kuliah?"
print(f"\n🎯 TES PENCARIAN FAISS: '{pertanyaan_tes}'")
hasil_pencarian = vector_store.similarity_search(pertanyaan_tes, k=2)

for i, doc in enumerate(hasil_pencarian):
    print(f"\n[Ranking {i+1}] Teks yang berhasil ditarik FAISS:")
    print(doc.page_content[:200] + " ... (dipotong)")