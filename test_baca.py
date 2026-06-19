import os
from dotenv import load_dotenv
# --- Hapus import Docx2txtLoader dan TextSplitter karena kita tidak membaca dokumen dari nol lagi ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
# --- Tambahkan path ke database vektor lokal ---
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakti_db_vektor")

PROMPT_TEMPLATE = """
Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip yang ramah dan solutif.
Tugasmu adalah menjawab pertanyaan mahasiswa penerima KIP-Kuliah Undip.

Gunakan HANYA informasi dari konteks berikut untuk menjawab. Jika jawaban tidak \
tersedia dalam konteks, sampaikan dengan jujur: "Maaf, SAKTI belum punya informasi \
soal itu, coba tanyakan ke pengurus Kamadiksi ya!" Jangan mengarang jawaban di luar konteks.

Konteks:
{context}

Pertanyaan: {question}

Jawaban:"""


def build_rag_chain():
    # 1. Siapkan mesin pencari vektor (Embeddings)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    
    # 2. LOAD DATABASE LOKAL (Tidak lagi membaca file DOCX dari awal)
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    # 3. Siapkan LLM dan Prompt
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # 4. Rangkai menjadi satu alur (Chain)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer(query: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(query)


if __name__ == "__main__":
    query = "Kak, untuk pengisian monev apakah wajib bagi penerima kipk di undip"

    print(f"Mahasiswa : {query}\n")
    result = answer(query)
    print(f"SAKTI :\n{result}")