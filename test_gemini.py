import os
from google import genai
from dotenv import load_dotenv

# 1. Membaca kunci rahasia dari file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Inisialisasi Klien Gemini (versi baru)
client = genai.Client(api_key=api_key)

# 3. Memberikan pertanyaan (Prompt)
print("Sedang mengirim pesan ke Gemini...")

# PERHATIKAN BARIS INI: Kita ganti modelnya ke generasi 2.0
response = client.models.generate_content(
    model='gemini-2.5-flash', 
    contents='Halo Gemini, saya Mustofa. Tolong beri saya satu kalimat motivasi pendek untuk menyelesaikan skripsi Teknik Komputer hari ini.'
)

# 4. Menampilkan jawaban
print("\n--- Jawaban Gemini ---")
print(response.text)