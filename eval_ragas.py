import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Siapkan LLM dan Embedding untuk "Juri" RAGAS
juri_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
juri_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

# 2. Contoh Data Pengujian SAKABOT
# Catatan: Di dunia nyata, Anda memasukkan 10-20 pertanyaan ke SAKABOT lalu mencatat hasilnya di sini.
# 2. Contoh Data Pengujian SAKABOT (Topik: Monev KIPK)
data_ujian = {
    "question": [
        "Kak, untuk pengisian monev apakah wajib bagi penerima kipk di undip?"
    ],
    "contexts": [
        # Ini simulasi teks yang berhasil ditarik FAISS dari dokumen KIPK Anda
        ["Sesuai dengan pedoman rektorat, seluruh mahasiswa Universitas Diponegoro yang berstatus sebagai penerima KIP Kuliah wajib melakukan pengisian formulir Monitoring dan Evaluasi (Monev) pada setiap semesternya."]
    ],
    "answer": [
        # Gunakan tanda kutip 3 (""") agar teks bisa dienter/multi-baris tanpa error
        """Halo! Iya betul sekali, Kak. Seluruh mahasiswa penerima KIP-K Undip **wajib** mengisi formulir Monev Ekonomi tanpa terkecuali. Kewajiban ini berlaku untuk semua angkatan, termasuk bagi penerima KIP-K susulan maupun pengganti. 
        Jadi, jangan lupa untuk segera diisi ya, Kak! Jika ada yang ingin ditanyakan lagi, SAKTI siap membantu."""
    ],
    "ground_truth": [
        # Kunci jawaban ideal/patokan dari Anda
        "Pengisian formulir Monev adalah wajib bagi seluruh mahasiswa penerima KIP Kuliah di Undip tanpa terkecuali."
    ]
}

# 3. Ubah format menjadi Dataset HuggingFace
dataset_eval = Dataset.from_dict(data_ujian)

# 4. Jalankan Proses Penilaian RAGAS
print("⏳ Memulai pengujian RAGAS, tunggu sebentar...")
hasil_evaluasi = evaluate(
    dataset=dataset_eval,
    metrics=[
        context_precision, # Seberapa akurat FAISS narik dokumen?
        context_recall,    # Apakah dokumen yang ditarik lengkap?
        faithfulness,      # Apakah jawaban bot sesuai fakta dokumen (tidak halusinasi)?
        answer_relevancy   # Apakah jawaban bot menjawab pertanyaan user?
    ],
    llm=juri_llm,
    embeddings=juri_embeddings
)

# 5. Tampilkan Hasil (Bentuk Tabel Pandas)
df_hasil = hasil_evaluasi.to_pandas()
print("\n=== SKOR RATA-RATA ===")
print(hasil_evaluasi)
print("\n=== DETAIL PER PERTANYAAN ===")
# Tampilkan seluruh tabel apa adanya
print(df_hasil.to_markdown())