import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# 1. SETUP AWAL
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

# 2. DATA INGESTION (Membaca & Memotong Dokumen)
print("1. Menyiapkan memori SAKTI...")
loader = Docx2txtLoader("data_pengetahuan/pedoman.docx")
potongan_dokumen = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

# 3. DATABASE VEKTOR (Otak Pencarian)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
db_vektor = FAISS.from_documents(potongan_dokumen, embeddings)

# Jadikan FAISS sebagai "Retriever" (Tukang Cari Buku), ambil 2 potongan paling relevan
pencari_konteks = db_vektor.as_retriever(search_kwargs={"k": 2})

# 4. LLM (Mulut Bot)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 5. PROMPT (Instruksi Kepribadian Bot)
template_sakti = """
Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip yang ramah, asyik, dan solutif.
Tugasmu adalah menjawab pertanyaan mahasiswa penerima KIP-Kuliah Undip.

Gunakan HANYA informasi dari konteks di bawah ini untuk menjawab pertanyaan. 
Jika di dalam konteks tidak ada jawabannya, bilang saja dengan jujur "Maaf, SAKTI belum punya informasi soal itu, coba tanyakan ke pengurus Kamadiksi ya!". 
JANGAN PERNAH mengarang jawaban sendiri di luar konteks.

Konteks Dokumen:
{context}

Pertanyaan Mahasiswa: {question}

Jawaban SAKTI:"""
prompt = PromptTemplate.from_template(template_sakti)

# 6. MERAKIT RAG CHAIN (Menyatukan Semuanya)
rag_chain = (
    {"context": pencari_konteks, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 7. MARI KITA TES NGOBROL DENGAN SAKTI!
# ==========================================
pertanyaan_mhs = "Kak, UKT aku masih ada tagihannya padahal aku KIPK tambahan, aku panik nih harus bayar atau ngga buat her-reg?"

print("\n" + "="*50)
print(f"🙋‍♂️ Mahasiswa : {pertanyaan_mhs}")
print("🤖 SAKTI sedang berpikir dan mengetik...")
print("="*50 + "\n")

# Menjalankan rantai RAG
jawaban_final = rag_chain.invoke(pertanyaan_mhs)

print(f"✨ SAKTI :\n{jawaban_final}\n")