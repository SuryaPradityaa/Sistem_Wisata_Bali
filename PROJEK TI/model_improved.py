import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset_wisata_bali.csv")

# =========================
# TEXT PREPROCESSING
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Dictionary sinonim bahasa Indonesia
synonyms = {
    "pantai": ["tepi laut", "pesisir", "pantay", "laut"],
    "gunung": ["bukit", "pegunungan", "puncak"],
    "berenang": ["mandi", "renang", "diving", "snorkeling"],
    "indah": ["bagus", "cantik", "menarik", "seru", "keren"],
    "liburan": ["libur", "traveling", "berlibur", "jalan jalan"],
    "tempat": ["lokasi", "destinasi", "spot", "area"],
    "bali": ["pulau bali", "bali pulau"],
    "murah": ["hemat", "ekonomis", "terjangkau", "minim"],
    "gratis": ["bebas", "tanpa bayar", "cuma cuma"],
    "terbaik": ["terpopuler", "terkenal", "paling bagus", "paling recommended"],
}

def preprocess_text(text):
    """Preprocessing text: lowercase, stemming, remove special chars"""
    # Lowercase
    text = text.lower()
    
    # Expand synonyms
    for key, values in synonyms.items():
        for syn in values:
            if syn in text:
                text = text.replace(syn, key)
    
    # Remove special characters tapi jaga spasi
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Stemming untuk setiap kata
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    
    return text

# =========================
# EXPANDED TRAINING DATA
# =========================
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
    ("pantai mana yang bagus", "pantai"),
    ("pantai terbaik bali", "pantai"),
    ("mau lihat laut", "pantai"),
    ("pantai untuk keluarga", "pantai"),
    ("pantai sepi dan tenang", "pantai"),
    ("diving di pantai", "pantai"),

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
    ("naik bukit", "alam"),
    ("panjat gunung", "alam"),
    ("mau di alam", "alam"),
    ("air terjun tersembunyi", "alam"),
    ("danau indah", "alam"),

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
    ("seni tradisional", "budaya"),
    ("mau belajar budaya", "budaya"),
    ("peninggalan sejarah", "budaya"),

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
    ("cari makan enak", "kuliner"),
    ("restoran bali", "kuliner"),
    ("makanan lokal", "kuliner"),
    ("pasar bali", "kuliner"),

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
    ("tempat asik", "hiburan"),
    ("main main", "hiburan"),

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
    ("hemat biaya", "murah"),
    ("gratis tiket", "murah"),

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
    ("tempat yang paling bagus", "terbaik"),
    ("favorit wisatawan", "terbaik"),

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
    ("bantu saya pilih", "bantuan"),
    ("gimana caranya", "bantuan"),
]

kalimat_train = [d[0] for d in training_data]
intent_train  = [d[1] for d in training_data]

# Preprocessing
kalimat_train_processed = [preprocess_text(k) for k in kalimat_train]

# =========================
# TRAIN MODEL IMPROVED
# =========================
model_chatbot = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),           # Unigram + Bigram
        min_df=1,
        max_df=0.95,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

# Fit dengan data yang sudah dipreprocess
model_chatbot.fit(kalimat_train_processed, intent_train)

# =========================
# FUNGSI PREDIKSI
# =========================
def predict_intent(user_input, confidence=False):
    """Predict intent dari input user"""
    processed = preprocess_text(user_input)
    intent = model_chatbot.predict([processed])[0]
    
    if confidence:
        probabilities = model_chatbot.predict_proba([processed])
        confidence_score = max(probabilities[0])
        return intent, confidence_score
    
    return intent

# =========================
# TEST MODEL
# =========================
if __name__ == "__main__":
    test_inputs = [
        "mau ke pantai yang indah",
        "rekomendasi tempat makan enak",
        "budget saya terbatas",
        "mau lihat alam",
        "pura mana yang terkenal",
        "halo bali guide",
        "pantai apa yang bagus",
    ]
    
    print("=== TEST MODEL IMPROVED ===\n")
    for user_input in test_inputs:
        intent, conf = predict_intent(user_input, confidence=True)
        print(f"Input: {user_input}")
        print(f"Intent: {intent} (confidence: {conf:.2f})\n")
