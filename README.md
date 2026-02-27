#  Dokumentasi Sistem RAG — Laboratorium Teknik Informatika ITS


---

## 1. Latar Belakang & Tujuan

### 1.1 Permasalahan

Informasi mengenai laboratorium di Departemen Teknik Informatika ITS tersebar di banyak halaman website dan tidak mudah diakses secara programatik. Pengguna baik mahasiswa, dosen, maupun pihak eksternal sering kesulitan mencari informasi seperti:

- Siapa kepala laboratorium tertentu?
- Dosen siapa saja yang tergabung dalam lab X?
- Mata kuliah apa yang ditawarkan di lab Y?
- Apa fasilitas yang tersedia di lab Z?

### 1.2 Solusi: RAG (Retrieval-Augmented Generation)

RAG adalah teknik yang menggabungkan dua komponen utama:

1. **Retriever**  mencari dokumen/chunk yang paling relevan dari knowledge base berdasarkan pertanyaan pengguna
2. **Generator (LLM)**  menggunakan dokumen relevan tersebut sebagai konteks untuk menghasilkan jawaban yang natural



---

## 2. Sumber & Struktur Data

### 2.1 Pengumpulan Data

1. Data dikumpulkan langsung dari website resmi ITS:

- `https://www.its.ac.id/informatika/id/laboratorium/`
- Masing-masing halaman per laboratorium
- Halaman daftar dosen: `https://www.its.ac.id/informatika/dosen-staff/daftar-dosen/`

2. Data dikumpulkan dengan interview Asisten Lab 

### 2.2 Format Output: JSON

Data disimpan dalam file `data_lab_informatika_its.json` dengan struktur:

```json
{
  "institusi": "Institut Teknologi Sepuluh Nopember (ITS) Surabaya",
  "departemen": "Departemen Teknik Informatika",
  "laboratorium": [
    {
      "id": "RPL",
      "nama": "Laboratorium Rekayasa Perangkat Lunak",
      "nama_inggris": "Laboratory of Software Engineering",
      "singkatan": "RPL",
      "lokasi": "...",
      "kepala_laboratorium": {
        "nama": "...",
        "email": "...",
        "inisial": "..."
      },
      "deskripsi": "...",
      "bidang_keahlian": ["...", "..."],
      "mata_kuliah": ["...", "..."],
      "fasilitas": { "komputer": ["..."] },
      "dosen_anggota": [{ "nama": "...", "jabatan": "...", "email": "..." }]
    }
  ]
}
```

**Alasan memilih JSON:**

- Format universal, mudah dibaca mesin maupun manusia
- Mendukung struktur hierarki (nested data)
- Langsung bisa di-parse Python tanpa library tambahan
- Ideal sebagai sumber data untuk pipeline RAG

### 2.3 Cakupan Data: 8 Laboratorium

| ID     | Nama Lab                                   | Kepala Lab                                      |
| ------ | ------------------------------------------ | ----------------------------------------------- |
| RPL    | Rekayasa Perangkat Lunak                   | Dr. Sarwosri, S.Kom., MT.                       |
| KBJ    | Komputasi Berbasis Jaringan                | Prof. Tohari Ahmad, S.Kom., M.IT., Ph.D.        |
| KCV    | Komputasi Cerdas dan Visi                  | Prof. Dr. Eng. Nanik Suciati, S.Kom., M.Kom.    |
| NETICS | Teknologi Jaringan & Keamanan Siber Cerdas | Royyana Muslim Ijtihadie, S.Kom., M.Kom., Ph.D. |
| GIGa   | Grafika, Interaksi, Gim dan Analitik       | Wijayanti Nurul Khotimah, S.Kom., M.Sc., Ph.D.  |
| AP     | Algoritma dan Pemrograman                  | Dr. Dwi Sunaryono, S.Kom., M.Kom.               |
| MCI    | Manajemen Cerdas Informasi                 | Ratih Nur Esti Anggraini, S.Kom., M.Sc., Ph.D.  |
| PKT    | Pemodelan dan Komputasi Terapan            | Dr. Bilqis Amaliah, S.Kom., M.Kom.              |

---

## 3. Arsitektur Sistem RAG

```
┌─────────────────────────────────────────────────────────┐
│                    TAHAP INDEXING                        │
│                                                         │
│  data_lab_informatika_its.json                          │
│         │                                               │
│         ▼                                               │
│   ┌─────────────┐     ┌──────────────────────────────┐  │
│   │  Chunking   │────▶│  Embedding Model             │  │
│   │  (5 chunk   │     │  paraphrase-multilingual-    │  │
│   │   per lab)  │     │  MiniLM-L12-v2               │  │
│   └─────────────┘     └──────────────┬───────────────┘  │
│                                      │                  │
│                                      ▼                  │
│                             ┌────────────────┐          │
│                             │  FAISS Vector  │          │
│                             │  Store (38 doc)│          │
│                             └────────────────┘          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    TAHAP INFERENCE                       │
│                                                         │
│  Pertanyaan User                                        │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │  Embedding  │────▶│  FAISS Similarity Search     │   │
│  │  Query      │     │  (top-k=4 dokumen)           │   │
│  └─────────────┘     └──────────────┬───────────────┘   │
│                                     │                   │
│                                     ▼                   │
│                        ┌────────────────────┐           │
│                        │  Prompt Template   │           │
│                        │  + Context + Query │           │
│                        └──────────┬─────────┘           │
│                                   │                     │
│                                   ▼                     │
│                      ┌───────────────────────┐          │
│                      │  LLM: Groq / Gemini   │          │
│                      │  llama-3.3-70b /      │          │
│                      │  gemini-2.5-flash     │          │
│                      └───────────┬───────────┘          │
│                                  │                      │
│                                  ▼                      │
│                            Jawaban ✅                   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Strategi Chunking

### 4.1 Mengapa Chunking Penting?

LLM memiliki **context window** (batas jumlah token). Jika seluruh data dimasukkan sekaligus, maka:

- Melebihi batas token
- LLM tidak bisa fokus pada informasi yang relevan
- Biaya komputasi meningkat

### 4.2 Strategi: Semantic Chunking per Aspek

Setiap laboratorium dipecah menjadi **5 chunk** berdasarkan aspek semantiknya:

| Chunk         | Isi                                 | Tujuan                      |
| ------------- | ----------------------------------- | --------------------------- |
| `profil`      | Nama, lokasi, kepala lab, deskripsi | Query tentang identitas lab |
| `dosen`       | Seluruh dosen anggota + email       | Query tentang SDM/kontak    |
| `mata_kuliah` | Mata kuliah + bidang keahlian       | Query tentang akademik      |
| `fasilitas`   | Perangkat keras + catatan khusus    | Query tentang infrastruktur |
| `asisten`     | Daftar asisten angkatan 2022 & 2023 | Query tentang asisten lab   |

**Total:** 8 lab × 5 chunk = **38 dokumen** di vector store _(6 lab memiliki data asisten; 2 lab tanpa data asisten = 4 chunk)_

**Keuntungan pendekatan ini dibanding character-based chunking:**

- Setiap chunk memiliki konteks yang koheren dan utuh
- Tidak ada informasi yang terpotong di tengah kalimat
- Retrieval lebih presisi karena setiap chunk mewakili topik spesifik
- Metadata (`lab`, `tipe`, `nama_lab`) memungkinkan filtering

---

## 5. Pemilihan Komponen & Alasannya

### 5.1 Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

| Aspek              | Detail                                      |
| ------------------ | ------------------------------------------- |
| **Ukuran model**   | ~118 MB (ringan)                            |
| **Dimensi vektor** | 384 dimensi                                 |
| **Bahasa**         | 50+ bahasa termasuk **Bahasa Indonesia** ✅ |
| **Sumber**         | Sentence-Transformers (HuggingFace)         |
| **Lisensi**        | Apache 2.0 (gratis, komersial)              |

**Mengapa model ini?**

Data laboratorium ITS **sepenuhnya berbahasa Indonesia**. Model embedding berbahasa Inggris (seperti `text-embedding-ada-002`) tidak akan memahami frasa seperti:

> _"Kepala Laboratorium"_, _"bidang keahlian"_, _"mata kuliah"_, _"rekayasa perangkat lunak"_

Model `paraphrase-multilingual-MiniLM-L12-v2` dilatih pada data **paralel multibahasa** sehingga memahami semantik Bahasa Indonesia dengan sangat baik, dan tetap ringan untuk dijalankan di Colab gratis.

---

### 5.2 Vector Store: FAISS

**FAISS** (Facebook AI Similarity Search) adalah library pencarian vektor yang dikembangkan oleh Meta AI Research.

**Mengapa FAISS?**

| Fitur                    | Keterangan                                                   |
| ------------------------ | ------------------------------------------------------------ |
| **Kecepatan**            | Optimasi C++, pencarian milidetik bahkan untuk jutaan vektor |
| **Gratis & Open Source** | Tidak perlu API key atau biaya                               |
| **Local**                | Berjalan di memory lokal, tidak perlu koneksi internet       |
| **Persistensi**          | Bisa disimpan ke disk dan dimuat ulang                       |
| **Integrasi LangChain**  | Dukungan penuh via `langchain_community.vectorstores`        |

Dengan hanya 38 dokumen, FAISS adalah pilihan **paling optimal** ,ringan, gratis, langsung jalan.

---

### 5.3 LLM: Perjalanan Eksperimen Model

dua model LLM diuji secara berurutan dalam proyek ini. Semua model menggunakan interface `llm` yang sama pada RAG chain penggantian model cukup dilakukan dengan mengubah konfigurasi satu sel.

#### Perjalanan Pemilihan Model

```
Groq: llama-3.3-70b-versatile
  ✅ Dicoba: 70B parameter, konteks 128K, multilingual
  ⚠️  Evaluasi: Jawaban lebih baik, namun gaya bahasa kurang natural
       → Coba model Gemini

Google Gemini: gemini-2.5-flash   ← MODEL FINAL ✅
  ✅ Dipilih: Jawaban paling natural, konteks besar, tersedia di API key aktif
```


> **Catatan:** Gemini menjadi pilihan saya karena jawaban yang diberikan sangat natural.

---

### 5.4 Framework: LangChain + LCEL

**LangChain** adalah framework orkestrasi untuk membangun aplikasi berbasis LLM.

**Mengapa LangChain?**

- Abstraksi yang konsisten untuk berbagai LLM, embedding, dan vector store
- Komponen modular: bisa ganti LLM atau embedding tanpa ubah kode lain
- Komunitas besar, dokumentasi lengkap

**Mengapa LCEL (LangChain Expression Language)?**

```python
# LCEL modern (digunakan di proyek ini)
rag_chain = (
    rag_chain_with_source
    | {"result": prompt | llm | StrOutputParser(),
       "source_documents": lambda x: x["context"]}
)
```

LCEL menggantikan `RetrievalQA` yang sudah **deprecated** sejak LangChain 0.2. Keunggulan LCEL:

| Fitur         | RetrievalQA (lama) | LCEL (baru)             |
| ------------- | ------------------ | ----------------------- |
| Streaming     | ❌                 | ✅                      |
| Async         | Terbatas           | ✅ Penuh                |
| Composability | ❌                 | ✅ Pipe operator (`\|`) |
| Debugging     | Sulit              | ✅ LangSmith native     |
| Status        | Deprecated         | ✅ Aktif dikembangkan   |

---

## 6. Pipeline Lengkap

```python
# 1. INSTALL
!pip install langchain langchain-community langchain-huggingface langchain-groq
!pip install faiss-cpu sentence-transformers huggingface_hub

# 2. LOAD DATA
data = json.load(open('data_lab_informatika_its.json'))

# 3. CHUNKING → 38 Document objects
docs = lab_to_documents(data)

# 4. EMBEDDING → Vektor 384 dimensi per chunk
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 5. VECTOR STORE → Index FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# 6. RETRIEVER → cari top-4 chunk paling relevan
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 7. LLM — pilih salah satu:
# Opsi A: Groq (gratis, cepat)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
# Opsi B: Gemini (natural, terbatas free tier)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# 8. RAG CHAIN → Retriever + Prompt + LLM
rag_chain = RunnableParallel(...) | {result: prompt | llm, source_documents: ...}

# 9. QUERY
result = rag_chain.invoke("Siapa kepala lab RPL?")
```

---

## 7. Konfigurasi Parameter & Justifikasi

| Parameter         | Nilai             | Alasan                                                                   |
| ----------------- | ----------------- | ------------------------------------------------------------------------ |
| `k=4` (retriever) | 4 dokumen         | Cukup konteks tanpa memenuhi prompt; 4 chunk = bisa mencakup 1 lab penuh |
| `temperature=0.1` | 0.1               | Jawaban faktual harus deterministik, bukan kreatif                       |
| `max_tokens=512`  | 512               | Cukup untuk jawaban deskriptif tanpa berlebihan                          |
| Chunk size        | 1 per aspek       | Semantic chunking, bukan character-based                                 |
| Similarity search | Cosine similarity | Default FAISS, cocok untuk vektor embedding                              |

---

## 8. Keterbatasan & Pengembangan Selanjutnya

### 8.1 Keterbatasan Saat Ini

| Keterbatasan                 | Dampak                                              |
| ---------------------------- | --------------------------------------------------- |
| Data statis (JSON)           | Jika web ITS diperbarui, data perlu di-scrape ulang |
| Tidak ada tahun berdiri lab  | Info founding year tidak tersedia di website resmi  |
| Nomor ruangan tidak tersedia | Website tidak mencantumkan nomor ruang spesifik     |
| LLM kadang "hallucinate"     | Jawaban bisa tidak akurat jika konteks ambigu       |

### 8.2 Potensi Pengembangan

- **Tambah data**: Penelitian terbaru, publikasi, tugas akhir mahasiswa
- **Multi-source RAG**: Gabungkan dengan data dari SINTA, Google Scholar
- **Web scraping otomatis**: Auto-update data JSON secara berkala (cron job)
- **Hybrid search**: Gabungkan dense retrieval (FAISS) + sparse (BM25) untuk akurasi lebih tinggi
- **Evaluation**: Implement RAGAs untuk mengukur faithfulness, context precision, dll.


---

## 9. Referensi & Sumber

| Komponen              | Sumber                                                                         |
| --------------------- | ------------------------------------------------------------------------------ |
| Data lab              | [its.ac.id/informatika](https://www.its.ac.id/informatika/id/laboratorium/)    |
| LangChain LCEL        | [python.langchain.com](https://python.langchain.com/docs/expression_language/) |
| FAISS                 | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| Sentence Transformers | [sbert.net](https://www.sbert.net/)                                            |
| Groq API              | [console.groq.com](https://console.groq.com/)                                  |
| Gemini API            | [aistudio.google](https://aistudio.google.com/)                                |
