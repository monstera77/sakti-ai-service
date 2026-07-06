import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Path dokumen dan folder database
DOC_PATH = "data_pengetahuan/pedoman.docx"
DB_PATH = "sakti_db_vektor"

def reindex_database():
    print("Mulai membaca dokumen...")
    # 2. Baca dokumen
    loader = Docx2txtLoader(DOC_PATH)
    documents = loader.load()

    # 3. Lakukan Chunking (Pemotongan Teks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Dokumen berhasil dipotong menjadi {len(chunks)} bagian.")

    # 4. Buat Embedding dan simpan ke FAISS (Timpa yang lama)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    print("Sedang membuat vektor dan menyimpan ke FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH) # <-- INI FUNGSI UNTUK MENYIMPANNYA
    print("Database FAISS berhasil diperbarui!")

if __name__ == "__main__":
    reindex_database()