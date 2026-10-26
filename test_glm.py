import os
from dotenv import load_dotenv

# 1. Ganti import Zhipu dengan OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY") 

EMBEDDING_MODEL = "models/gemini-embedding-001"
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
    # Embeddings tetap Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. TRIK JITU: Gunakan modul OpenAI, tapi arahkan base_url ke Zhipu
    llm = ChatOpenAI(
        model_name="glm-4-flash", 
        api_key=ZHIPUAI_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/", # URL peladen Zhipu
        temperature=0.3
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

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
    query = "kapan pencairan KIP-Kuliah Undip semester ganjil?"

    print(f"Mahasiswa : {query}\n")
    result = answer(query)
    print(f"SAKTI (GLM-4) :\n{result}")