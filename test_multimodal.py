import os
import base64
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# 1. SETUP AWAL
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

# Fungsi untuk mengubah gambar menjadi format Base64 (agar bisa dikirim ke otak Gemini)
def baca_gambar_base64(path_gambar):
    with open(path_gambar, "rb") as file_gambar:
        return base64.b64encode(file_gambar.read()).decode('utf-8')

# 2. PERSIAPAN DATA TEKS & FAISS (Memori SAKTI)
print("1. Membangun ulang memori teks SAKTI...")
loader = Docx2txtLoader("data_pengetahuan/pedoman.docx")
potongan = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
db_vektor = FAISS.from_documents(potongan, embeddings)

# 3. PERSIAPAN MULTIMODAL (Mata & Mulut SAKTI)
# Kita pakai Gemini 2.5 Flash yang punya fitur bawaan membaca gambar
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# ==========================================
# 4. SIMULASI KASUS MAHASISWA (TEKS + GAMBAR)
# ==========================================
pertanyaan_mhs = "izin bertanya kak ini saya sudah melakukan pengisian dan submit berkas namun mepet dengan jadwal yang ditentukan apakah aman atau bagaimana ya kak karena yang tertera di situ sudah tutup kak?"
path_gambar_mhs = "data_pengetahuan/telat.png" # <-- PASTIKAN NAMA FILE DAN LOKASINYA BENAR!

print(f"\n🙋‍♂️ Mahasiswa : {pertanyaan_mhs}")
print(f"📸 Melampirkan Gambar : {path_gambar_mhs}")
print("🤖 SAKTI sedang melihat gambar dan mencari buku panduan...")
print("="*60)

# 5. PENCARIAN BUKU PANDUAN (RAG)
# SAKTI mencari dokumen yang relevan dengan keluhan teks mahasiswa
hasil_pencarian = db_vektor.similarity_search(pertanyaan_mhs, k=2)
konteks_dokumen = "\n".join([doc.page_content for doc in hasil_pencarian])

# 6. MERAKIT PESAN MULTIMODAL
# Kita ubah gambar jadi sandi base64
gambar_base64 = baca_gambar_base64(path_gambar_mhs)

# Format pesan khusus yang menggabungkan Prompt, Teks FAISS, dan Gambar
pesan_sakti = HumanMessage(
    content=[
        {
            "type": "text", 
            "text": f"""Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip.
            Tugasmu adalah menganalisis gambar screenshot keluhan mahasiswa dan menjawab pertanyaannya HANYA berdasarkan dokumen panduan berikut.
            
            Dokumen Panduan (Konteks RAG):
            {konteks_dokumen}
            
            Pertanyaan Mahasiswa: {pertanyaan_mhs}
            
            Instruksi: Jawablah dengan ramah dan solutif ala kakak tingkat. Beritahu mahasiswa apa yang kamu lihat di screenshot tersebut, lalu berikan solusi konkret berdasarkan Dokumen Panduan."""
        },
        {
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{gambar_base64}"}
        }
    ]
)

# 7. EKSEKUSI!
jawaban_final = llm.invoke([pesan_sakti])
print(f"\n✨ SAKTI :\n{jawaban_final.content}\n")