import os
import base64
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage


load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

KNOWLEDGE_DOC = "data_pengetahuan/pedoman.docx"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_vector_store(doc_path: str) -> FAISS:
    loader = Docx2txtLoader(doc_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vector_store: FAISS, query: str, k: int = 2) -> str:
    results = vector_store.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in results)


def build_message(query: str, context: str, image_path: str) -> HumanMessage:
    image_b64 = encode_image(image_path)

    system_prompt = f"""Kamu adalah SAKTI, Asisten Virtual Kamadiksi Undip.
Tugasmu adalah menganalisis screenshot yang dilampirkan mahasiswa dan menjawab pertanyaannya berdasarkan dokumen panduan berikut.

Dokumen Panduan:
{context}

Pertanyaan Mahasiswa: {query}

Instruksi: Jawablah dengan ramah dan solutif. Jelaskan apa yang terlihat di screenshot, lalu berikan solusi konkret sesuai panduan."""

    return HumanMessage(
        content=[
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    )


def answer(query: str, image_path: str) -> str:
    vector_store = build_vector_store(KNOWLEDGE_DOC)
    context = retrieve_context(vector_store, query)

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    message = build_message(query, context, image_path)

    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    query = (
        "izin bertanya kak ini saya sudah melakukan pengisian dan submit berkas "
        "namun mepet dengan jadwal yang ditentukan apakah aman atau bagaimana ya kak "
        "karena yang tertera di situ sudah tutup kak?"
    )
    image_path = "data_pengetahuan/telat.png"

    print(f"Mahasiswa : {query}")
    print(f"Lampiran  : {image_path}\n")

    result = answer(query, image_path)
    print(f"SAKTI :\n{result}")