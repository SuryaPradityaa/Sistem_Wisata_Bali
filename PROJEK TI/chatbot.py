import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset_wisata_bali.csv")

# =========================
# TRAINING DATA (Intent)
# =========================
# Data latih: pasangan (kalimat_user, intent)
training_data = [
    # --- PANTAI ---
    ("saya mau ke pantai", "pantai"),
    ("rekomendasi pantai di bali", "pantai"),
    ("pantai yang bagus", "pantai"),
    ("wisata pantai", "pantai"),
    ("mau liburan ke tepi laut", "pantai"),
    ("pantai yang sepi", "pantai"),
    ("ingin berenang di pantai", "pantai"),
    ("pantai indah bali", "pantai"),
    ("ke pantai yuk", "pantai"),
    ("mau snorkeling", "pantai"),
    ("mau surfing", "pantai"),
    ("pantai pasir putih", "pantai"),
    ("pantai romantis", "pantai"),
    ("sunset di pantai", "pantai"),

    # --- ALAM ---
    ("wisata alam", "alam"),
    ("mau ke gunung", "alam"),
    ("air terjun bali", "alam"),
    ("trekking di bali", "alam"),
    ("hiking bali", "alam"),
    ("wisata pegunungan", "alam"),
    ("mau ke hutan", "alam"),
    ("pemandangan alam indah", "alam"),
    ("danau bali", "alam"),
    ("air terjun yang bagus", "alam"),
    ("wisata outbound", "alam"),
    ("alam bali yang asri", "alam"),
    ("mau camping", "alam"),
    ("sawah terasering", "alam"),

    # --- BUDAYA ---
    ("wisata budaya", "budaya"),
    ("mau ke pura", "budaya"),
    ("tempat bersejarah bali", "budaya"),
    ("wisata tradisional", "budaya"),
    ("mau lihat tari kecak", "budaya"),
    ("museum bali", "budaya"),
    ("pura besar bali", "budaya"),
    ("kebudayaan bali", "budaya"),
    ("mau ke ubud", "budaya"),
    ("seni budaya bali", "budaya"),
    ("mau ke desa adat", "budaya"),
    ("temple bali", "budaya"),
    ("mau foto di pura", "budaya"),

    # --- KULINER ---
    ("wisata kuliner", "kuliner"),
    ("mau belanja", "kuliner"),
    ("pasar oleh oleh", "kuliner"),
    ("makanan khas bali", "kuliner"),
    ("tempat makan bali", "kuliner"),
    ("belanja souvenir", "kuliner"),
    ("pasar tradisional", "kuliner"),
    ("oleh oleh bali", "kuliner"),
    ("mau cari makan", "kuliner"),
    ("kuliner enak bali", "kuliner"),
    ("tempat belanja murah", "kuliner"),

    # --- HIBURAN ---
    ("wisata hiburan", "hiburan"),
    ("taman bermain", "hiburan"),
    ("tempat seru", "hiburan"),
    ("mau seru seruan", "hiburan"),
    ("wahana permainan", "hiburan"),
    ("liburan keluarga", "hiburan"),
    ("bawa anak main", "hiburan"),
    ("taman hiburan bali", "hiburan"),
    ("waterpark bali", "hiburan"),
    ("mau yang seru", "hiburan"),

    # --- MURAH ---
    ("wisata murah", "murah"),
    ("tempat gratis bali", "murah"),
    ("liburan hemat", "murah"),
    ("tidak punya banyak uang", "murah"),
    ("budget terbatas", "murah"),
    ("tiket murah", "murah"),
    ("wisata gratis", "murah"),
    ("tempat murah meriah", "murah"),
    ("liburan dengan budget minim", "murah"),

    # --- RATING TINGGI ---
    ("tempat terbaik di bali", "terbaik"),
    ("wisata paling populer", "terbaik"),
    ("rekomendasi terbaik", "terbaik"),
    ("tempat yang bagus banget", "terbaik"),
    ("rating tertinggi", "terbaik"),
    ("tempat terkenal bali", "terbaik"),
    ("yang paling recommended", "terbaik"),
    ("destinasi terpopuler", "terbaik"),
    ("wisata hits bali", "terbaik"),

    # --- LOKASI ---
    ("wisata di badung", "lokasi_badung"),
    ("tempat di badung", "lokasi_badung"),
    ("seminyak kuta", "lokasi_badung"),
    ("wisata di ubud", "lokasi_gianyar"),
    ("tempat di gianyar", "lokasi_gianyar"),
    ("wisata di denpasar", "lokasi_denpasar"),
    ("tempat di denpasar", "lokasi_denpasar"),
    ("wisata di karangasem", "lokasi_karangasem"),
    ("wisata di buleleng", "lokasi_buleleng"),
    ("wisata di tabanan", "lokasi_tabanan"),
    ("wisata di bangli", "lokasi_bangli"),
    ("wisata di klungkung", "lokasi_klungkung"),
    ("wisata di jembrana", "lokasi_jembrana"),

    # --- SALAM / UMUM ---
    ("halo", "salam"),
    ("hai", "salam"),
    ("hello", "salam"),
    ("selamat pagi", "salam"),
    ("selamat siang", "salam"),
    ("selamat malam", "salam"),
    ("hei bali guide", "salam"),
    ("apa kabar", "salam"),
    ("help", "bantuan"),
    ("tolong bantu", "bantuan"),
    ("bisa bantu saya", "bantuan"),
    ("mau liburan ke bali", "bantuan"),
    ("tidak tahu mau kemana", "bantuan"),
    ("bingung mau ke mana", "bantuan"),
    ("ada rekomendasi", "bantuan"),
]

kalimat_train = [d[0] for d in training_data]
intent_train  = [d[1] for d in training_data]

# =========================
# TRAIN MODEL
# =========================
model_chatbot = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf',   MultinomialNB(alpha=0.1))
])
model_chatbot.fit(kalimat_train, intent_train)

# =========================
# FUNGSI BANTU
# =========================
def format_tempat(row, idx=None):
    prefix = f"{idx}. " if idx else "• "
    return (
        f"{prefix}**{row['nama_tempat']}**\n"
        f"   📍 {row['kabupaten']} | 🏷️ {row['kategori']} | ⭐ {row['rating']}\n"
        f"   💰 Tiket WNI: Rp{int(row['harga_tiket_wni']):,} | ⏰ {row['jam_buka']}\n"
        f"   🎯 Aktivitas: {row['aktivitas']}\n"
        f"   📝 {str(row['deskripsi'])[:120]}...\n"
    )

def get_rekomendasi_by_kategori(kategori, top=4):
    hasil = df[df['kategori'].str.lower() == kategori.lower()]
    hasil = hasil.sort_values('rating', ascending=False).head(top)
    return hasil

def get_rekomendasi_by_kabupaten(kabupaten, top=4):
    hasil = df[df['kabupaten'].str.lower().str.contains(kabupaten.lower())]
    hasil = hasil.sort_values('rating', ascending=False).head(top)
    return hasil

def get_rekomendasi_murah(top=4):
    hasil = df[df['harga_tiket_wni'] <= 20000]
    hasil = hasil.sort_values('rating', ascending=False).head(top)
    return hasil

def get_rekomendasi_terbaik(top=5):
    return df.sort_values('rating', ascending=False).head(top)

def cek_keyword_lokasi(teks):
    lokasi_map = {
        "badung":      "Badung",
        "seminyak":    "Badung",
        "kuta":        "Badung",
        "ubud":        "Gianyar",
        "gianyar":     "Gianyar",
        "denpasar":    "Denpasar",
        "karangasem":  "Karangasem",
        "buleleng":    "Buleleng",
        "singaraja":   "Buleleng",
        "tabanan":     "Tabanan",
        "bangli":      "Bangli",
        "klungkung":   "Klungkung",
        "jembrana":    "Jembrana",
    }
    teks = teks.lower()
    for keyword, kabupaten in lokasi_map.items():
        if keyword in teks:
            return kabupaten
    return None

# =========================
# FUNGSI UTAMA CHATBOT
# =========================
def chat_ml(pesan_user: str, riwayat: list = None) -> str:
    teks = pesan_user.strip()
    teks_lower = teks.lower()

    # Prediksi intent
    intent = model_chatbot.predict([teks])[0]
    proba  = model_chatbot.predict_proba([teks]).max()

    # Jika confidence rendah, fallback ke pencarian keyword
    if proba < 0.25:
        intent = "bantuan"

    # Cek lokasi spesifik dari teks user
    lokasi = cek_keyword_lokasi(teks_lower)

    # ── SALAM ──
    if intent == "salam":
        return (
            "Halo! 👋 Saya **BaliGuide**, asisten wisata Bali kamu.\n\n"
            "Saya bisa membantu kamu menemukan:\n"
            "🏖️ Pantai • 🌿 Alam • 🏛️ Budaya • 🍜 Kuliner • 🎡 Hiburan\n\n"
            "Ceritakan saja mau wisata apa atau ke mana, saya siap bantu! 😊"
        )

    # ── BANTUAN / UMUM ──
    if intent == "bantuan":
        return (
            "Tentu! Saya siap bantu kamu liburan di Bali 🌴\n\n"
            "Coba ceritakan preferensi kamu, misalnya:\n"
            "• *\"Saya mau ke pantai yang sepi\"*\n"
            "• *\"Ada wisata budaya di Ubud?\"*\n"
            "• *\"Rekomendasikan tempat murah di Bali\"*\n"
            "• *\"Tempat terbaik di Bali apa?\"*\n\n"
            "Atau sebutkan kabupaten yang ingin kamu kunjungi! 😊"
        )

    # ── PANTAI ──
    if intent == "pantai":
        if lokasi:
            hasil = df[(df['kategori'] == 'Pantai') &
                       (df['kabupaten'].str.lower() == lokasi.lower())]
            hasil = hasil.sort_values('rating', ascending=False).head(4)
            judul = f"🏖️ Pantai di **{lokasi}**"
        else:
            hasil = get_rekomendasi_by_kategori("Pantai")
            judul = "🏖️ Rekomendasi Pantai di Bali"

        if hasil.empty:
            return f"Maaf, tidak ada pantai yang ditemukan di {lokasi}. Coba kabupaten lain?"

        teks_hasil = f"{judul}:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Datang pagi hari untuk menghindari keramaian!*"
        return teks_hasil

    # ── ALAM ──
    if intent == "alam":
        if lokasi:
            hasil = df[(df['kategori'] == 'Alam') &
                       (df['kabupaten'].str.lower() == lokasi.lower())]
            hasil = hasil.sort_values('rating', ascending=False).head(4)
            judul = f"🌿 Wisata Alam di **{lokasi}**"
        else:
            hasil = get_rekomendasi_by_kategori("Alam")
            judul = "🌿 Wisata Alam di Bali"

        if hasil.empty:
            return "Maaf, tidak ada wisata alam yang ditemukan di lokasi tersebut."

        teks_hasil = f"{judul}:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Gunakan alas kaki yang nyaman untuk trekking!*"
        return teks_hasil

    # ── BUDAYA ──
    if intent == "budaya":
        if lokasi:
            hasil = df[(df['kategori'] == 'Budaya') &
                       (df['kabupaten'].str.lower() == lokasi.lower())]
            hasil = hasil.sort_values('rating', ascending=False).head(4)
            judul = f"🏛️ Wisata Budaya di **{lokasi}**"
        else:
            hasil = get_rekomendasi_by_kategori("Budaya")
            judul = "🏛️ Wisata Budaya di Bali"

        if hasil.empty:
            return "Maaf, tidak ada wisata budaya di lokasi tersebut."

        teks_hasil = f"{judul}:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Kenakan pakaian sopan saat mengunjungi pura!*"
        return teks_hasil

    # ── KULINER ──
    if intent == "kuliner":
        hasil = get_rekomendasi_by_kategori("Kuliner & Belanja")
        if hasil.empty:
            return "Maaf, data kuliner tidak ditemukan."

        teks_hasil = "🍜 Wisata Kuliner & Belanja di Bali:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Tawar menawar adalah hal biasa di pasar tradisional!*"
        return teks_hasil

    # ── HIBURAN ──
    if intent == "hiburan":
        hasil = get_rekomendasi_by_kategori("Taman Hiburan")
        if hasil.empty:
            return "Maaf, data hiburan tidak ditemukan."

        teks_hasil = "🎡 Wisata Hiburan di Bali:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Cek jam operasional sebelum berangkat!*"
        return teks_hasil

    # ── MURAH ──
    if intent == "murah":
        hasil = get_rekomendasi_murah()
        if hasil.empty:
            return "Semua tempat wisata tersedia dengan harga terjangkau!"

        teks_hasil = "💰 Wisata Murah & Hemat di Bali (Tiket ≤ Rp20.000):\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Tips: Bawa bekal sendiri untuk hemat lebih banyak!*"
        return teks_hasil

    # ── TERBAIK ──
    if intent == "terbaik":
        hasil = get_rekomendasi_terbaik()
        teks_hasil = "⭐ Top 5 Tempat Wisata Terbaik di Bali:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        teks_hasil += "\n💡 *Semua tempat di atas memiliki rating tertinggi dari pengunjung!*"
        return teks_hasil

    # ── LOKASI SPESIFIK ──
    if intent.startswith("lokasi_") or lokasi:
        kab = lokasi or intent.replace("lokasi_", "").capitalize()
        hasil = get_rekomendasi_by_kabupaten(kab)
        if hasil.empty:
            return f"Maaf, tidak ada tempat wisata yang ditemukan di **{kab}**."

        teks_hasil = f"📍 Wisata di **{kab}**:\n\n"
        for i, (_, row) in enumerate(hasil.iterrows(), 1):
            teks_hasil += format_tempat(row, i)
        return teks_hasil

    # ── FALLBACK ──
    return (
        "Hmm, saya kurang mengerti pertanyaanmu. 😅\n\n"
        "Coba tanyakan seperti ini:\n"
        "• *\"Rekomendasikan pantai di Badung\"*\n"
        "• *\"Wisata alam di Bali\"*\n"
        "• *\"Tempat wisata murah\"*\n"
        "• *\"Wisata terbaik di Bali\"*"
    )


# Alias agar app.py tetap bisa import dengan nama chat_gemini
def chat_gemini(pesan_user: str, riwayat: list = None) -> str:
    return chat_ml(pesan_user, riwayat)