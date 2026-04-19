import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

KNOWLEDGE_DOC = "data_pengetahuan/pedoman.docx"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

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
    loader = Docx2txtLoader(KNOWLEDGE_DOC)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(loader.load())

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
        search_kwargs={"k": 2}
    )

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
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
    query = "Kak, UKT aku masih ada tagihannya padahal aku KIPK tambahan, aku panik nih harus bayar atau ngga buat her-reg?"

    print(f"Mahasiswa : {query}\n")
    result = answer(query)
    print(f"SAKTI :\n{result}")